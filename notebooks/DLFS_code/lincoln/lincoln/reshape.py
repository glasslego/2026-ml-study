"""텐서 모양을 바꾸는 연산 모음."""

from numpy import ndarray

from .base import Operation


class Flatten(Operation):
    """배치 차원은 유지한 채 나머지 차원을 한 줄로 펴는 연산."""

    def __init__(self):
        super().__init__()

    def _output(self, inference: bool = False) -> ndarray:
        # 합성곱 출력 [batch, channel, height, width]를
        # 완전연결층 입력 [batch, features]로 바꿀 때 자주 사용한다.
        return self.input_.reshape(self.input_.shape[0], -1)

    def _input_grad(self, output_grad: ndarray) -> ndarray:
        # 역전파에서는 flatten 이전의 원래 shape로 다시 되돌려야 한다.
        return output_grad.reshape(self.input_.shape)
