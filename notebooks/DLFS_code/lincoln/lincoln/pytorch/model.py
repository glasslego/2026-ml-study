"""PyTorch 모델 공통 베이스 클래스."""

from typing import Tuple

from torch import nn, Tensor


class PyTorchModel(nn.Module):
    """Trainer가 기대하는 인터페이스를 문서화하는 추상 모델."""

    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: Tensor) -> Tuple[Tensor]:
        # 이 코드베이스의 trainer는 model(X)[0] 형태를 기대하므로
        # 하위 모델도 tuple을 반환하는 패턴을 따른다.
        raise NotImplementedError()
