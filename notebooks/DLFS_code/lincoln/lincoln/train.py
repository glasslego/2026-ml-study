"""NumPy 기반 신경망 학습 루프."""

from copy import deepcopy
from typing import Tuple

import numpy as np
from numpy import ndarray

from .network import NeuralNetwork
from .optimizers import Optimizer
from lincoln.utils.np_utils import permute_data


class Trainer(object):
    """네트워크와 optimizer를 묶어 학습을 진행하는 도우미."""

    def __init__(self, net: NeuralNetwork, optim: Optimizer) -> None:
        self.net = net
        self.optim = optim
        self.best_loss = 1e9

        # optimizer는 스스로 파라미터를 알 수 없으므로 trainer가 net 참조를 주입한다.
        setattr(self.optim, "net", self.net)

    def fit(
        self,
        X_train: ndarray,
        y_train: ndarray,
        X_test: ndarray,
        y_test: ndarray,
        epochs: int = 100,
        eval_every: int = 10,
        batch_size: int = 32,
        seed: int = 1,
        single_output: bool = False,
        restart: bool = True,
        early_stopping: bool = True,
        conv_testing: bool = False,
    ) -> None:
        """여러 epoch에 걸쳐 모델을 학습한다."""

        setattr(self.optim, "max_epochs", epochs)
        self.optim._setup_decay()

        np.random.seed(seed)
        if restart:
            # lazy setup을 쓰는 layer는 첫 forward에서 파라미터를 다시 초기화해야 한다.
            for layer in self.net.layers:
                layer.first = True

            self.best_loss = 1e9

        for e in range(epochs):
            if (e + 1) % eval_every == 0:
                # early stopping에서 성능이 악화되면 직전 좋은 모델로 되돌리기 위해 저장한다.
                last_model = deepcopy(self.net)

            X_train, y_train = permute_data(X_train, y_train)

            batch_generator = self.generate_batches(X_train, y_train, batch_size)

            for ii, (X_batch, y_batch) in enumerate(batch_generator):
                self.net.train_batch(X_batch, y_batch)

                self.optim.step()

                if conv_testing:
                    if ii % 10 == 0:
                        test_preds = self.net.forward(X_batch, inference=True)
                        batch_loss = self.net.loss.forward(test_preds, y_batch)
                        print(ii, "배치 학습 후 손실값은 ", batch_loss)

                    if ii % 100 == 0 and ii > 0:
                        print(
                            ii,
                            "배치 학습 후 검증 데이터에 대한 정확도: ",
                            f"""{np.equal(
                                np.argmax(self.net.forward(X_test, inference=True), axis=1),
                                np.argmax(y_test, axis=1),
                            ).sum() * 100.0 / X_test.shape[0]:.2f}%""",
                        )

            if (e + 1) % eval_every == 0:
                test_preds = self.net.forward(X_test, inference=True)
                loss = self.net.loss.forward(test_preds, y_test)

                if early_stopping:
                    if loss < self.best_loss:
                        print(f"{e+1}에폭에서 검증 데이터에 대한 손실값: {loss:.3f}")
                        self.best_loss = loss
                    else:
                        print()
                        print(
                            f"{e+1}에폭에서 손실값이 증가했다. 마지막으로 측정한 손실값은 {e+1-eval_every}",
                            f"에폭까지 학습된 모델에서 계산된 {self.best_loss:.3f}이다.",
                        )
                        self.net = last_model

                        # net 참조를 바꿨으므로 optimizer도 새 객체를 바라보게 다시 연결한다.
                        setattr(self.optim, "net", self.net)
                        break
                else:
                    print(f"{e+1}에폭에서 검증 데이터에 대한 손실값: {loss:.3f}")

            if self.optim.final_lr:
                self.optim._decay_lr()

    def generate_batches(
        self,
        X: ndarray,
        y: ndarray,
        size: int = 32,
    ) -> Tuple[ndarray]:
        """데이터를 고정 길이 배치로 잘라 generator로 반환한다."""

        assert X.shape[0] == y.shape[0], """
        features and target must have the same number of rows, instead
        features has {0} and target has {1}
        """.format(X.shape[0], y.shape[0])

        N = X.shape[0]

        for ii in range(0, N, size):
            X_batch, y_batch = X[ii : ii + size], y[ii : ii + size]

            yield X_batch, y_batch
