"""PyTorch 버전 layer 래퍼들."""

import torch.nn as nn
from torch import Tensor


def inference_mode(m: nn.Module):
    """하위 모듈을 추론 모드로 전환한다."""

    m.eval()


class PyTorchLayer(nn.Module):
    """책의 NumPy layer 인터페이스와 비슷한 느낌을 주는 추상 클래스."""

    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: Tensor, inference: bool = False) -> Tensor:
        raise NotImplementedError()


class DenseLayer(PyTorchLayer):
    """`nn.Linear`를 감싼 간단한 완전연결층."""

    def __init__(
        self,
        input_size: int,
        neurons: int,
        dropout: float = 1.0,
        activation: nn.Module = None,
    ) -> None:
        super().__init__()
        self.linear = nn.Linear(input_size, neurons)
        self.activation = activation
        if dropout < 1.0:
            # 책 코드의 dropout은 keep probability이고,
            # PyTorch nn.Dropout은 drop probability를 받으므로 1 - dropout을 넘긴다.
            self.dropout = nn.Dropout(1 - dropout)

    def forward(self, x: Tensor, inference: bool = False) -> Tensor:
        if inference:
            self.apply(inference_mode)

        x = self.linear(x)
        if self.activation:
            x = self.activation(x)
        if hasattr(self, "dropout"):
            x = self.dropout(x)

        return x
