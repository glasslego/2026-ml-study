"""PyTorch 모델 입력 전처리기."""

from torch import Tensor


class PyTorchPreprocessor:
    """입력 텐서 전처리 인터페이스."""

    def __init__(self):
        pass

    def transform(self, x: Tensor) -> Tensor:
        raise NotImplementedError()


class ConvNetPreprocessor(PyTorchPreprocessor):
    """이미지 텐서를 CNN이 기대하는 축 순서로 바꾼다."""

    def __init__(self):
        pass

    def transform(self, x: Tensor) -> Tensor:
        # 많은 데이터셋은 [batch, height, width, channel] 순서를 쓰지만
        # PyTorch CNN은 [batch, channel, height, width]를 기대한다.
        return x.permute(0, 3, 1, 2)
