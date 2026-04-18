"""PyTorch 기반 학습 루프."""

from typing import Tuple

import torch
from torch import Tensor
from torch.nn.modules.loss import _Loss
from torch.optim import Optimizer
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader

from .model import PyTorchModel
from .utils import permute_data


class PyTorchTrainer(object):
    """PyTorch 모델 학습을 단순한 인터페이스로 감싼 도우미."""

    def __init__(self, model: PyTorchModel, optim: Optimizer, criterion: _Loss):
        self.model = model
        self.optim = optim
        self.loss = criterion
        self._check_optim_net_aligned()

    def _check_optim_net_aligned(self):
        """optimizer가 실제 현재 모델 파라미터를 보고 있는지 확인한다."""

        assert self.optim.param_groups[0]["params"] == list(self.model.parameters())

    def _generate_batches(
        self,
        X: Tensor,
        y: Tensor,
        size: int = 32,
    ) -> Tuple[Tensor]:
        """텐서에서 mini-batch를 순서대로 잘라낸다."""

        N = X.shape[0]

        for ii in range(0, N, size):
            X_batch, y_batch = X[ii : ii + size], y[ii : ii + size]

            yield X_batch, y_batch

    def fit(
        self,
        X_train: Tensor = None,
        y_train: Tensor = None,
        X_test: Tensor = None,
        y_test: Tensor = None,
        train_dataloader: DataLoader = None,
        test_dataloader: DataLoader = None,
        epochs: int = 100,
        eval_every: int = 10,
        batch_size: int = 32,
        final_lr_exp: int = None,
    ):
        """텐서 입력 모드 또는 DataLoader 모드로 모델을 학습한다."""

        init_lr = self.optim.param_groups[0]["lr"]
        if final_lr_exp:
            decay = (final_lr_exp / init_lr) ** (1.0 / (epochs + 1))
            scheduler = lr_scheduler.ExponentialLR(self.optim, gamma=decay)
        for e in range(epochs):
            if not train_dataloader:
                X_train, y_train = permute_data(X_train, y_train)

                batch_generator = self._generate_batches(X_train, y_train, batch_size)

                self.model.train()

                for ii, (X_batch, y_batch) in enumerate(batch_generator):
                    # PyTorch의 표준 학습 순서:
                    # zero_grad -> forward -> loss -> backward -> optimizer.step
                    self.optim.zero_grad()

                    output = self.model(X_batch)[0]

                    loss = self.loss(output, y_batch)
                    loss.backward()
                    self.optim.step()

                if e % eval_every == 0:
                    with torch.no_grad():
                        self.model.eval()
                        output = self.model(X_test)[0]
                        loss = self.loss(output, y_test)
                        print(e + 1, " 에폭 학습한 후의 손실은 ", loss.item())

            else:
                for X_batch, y_batch in train_dataloader:
                    self.optim.zero_grad()

                    output = self.model(X_batch)[0]

                    loss = self.loss(output, y_batch)
                    loss.backward()
                    self.optim.step()

                if e % eval_every == 0:
                    with torch.no_grad():
                        self.model.eval()
                        losses = []
                        for X_batch, y_batch in test_dataloader:
                            output = self.model(X_batch)[0]
                            loss = self.loss(output, y_batch)
                            losses.append(loss.item())
                        print(
                            e,
                            " 에폭 학습한 후의 손실은 ",
                            round(torch.Tensor(losses).mean().item(), 4),
                        )

            if final_lr_exp:
                # PyTorch 권장 순서에 맞춰 optimizer.step 이후 epoch 끝에서 scheduler를 갱신한다.
                scheduler.step()
