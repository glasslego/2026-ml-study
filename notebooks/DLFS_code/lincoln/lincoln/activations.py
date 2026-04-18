"""활성화 함수와 그 미분을 NumPy로 구현한 모듈."""

import numpy as np
from numpy import ndarray

from .base import Operation


class Linear(Operation):
    """입력을 그대로 통과시키는 항등 활성화 함수."""

    def __init__(self) -> None:
        super().__init__()

    def _output(self, inference: bool) -> ndarray:
        # 선형 활성화는 아무 변형도 하지 않으므로 입력을 그대로 반환한다.
        return self.input_

    def _input_grad(self, output_grad: ndarray) -> ndarray:
        # y = x 의 도함수는 1이므로 gradient도 그대로 흘러간다.
        return output_grad


class Sigmoid(Operation):
    """시그모이드 활성화 함수."""

    def __init__(self) -> None:
        super().__init__()

    def _output(self, inference: bool) -> ndarray:
        return 1.0 / (1.0 + np.exp(-1.0 * self.input_))

    def _input_grad(self, output_grad: ndarray) -> ndarray:
        # sigmoid'(x) = sigmoid(x) * (1 - sigmoid(x))
        # 순전파 결과 self.output을 재사용하면 x를 다시 계산하지 않아도 된다.
        sigmoid_backward = self.output * (1.0 - self.output)
        input_grad = sigmoid_backward * output_grad
        return input_grad


class Tanh(Operation):
    """하이퍼볼릭 탄젠트 활성화 함수."""

    def __init__(self) -> None:
        super().__init__()

    def _output(self, inference: bool) -> ndarray:
        return np.tanh(self.input_)

    def _input_grad(self, output_grad: ndarray) -> ndarray:
        # tanh'(x) = 1 - tanh(x)^2
        return output_grad * (1 - self.output * self.output)


class ReLU(Operation):
    """ReLU(Rectified Linear Unit) 활성화 함수."""

    def __init__(self) -> None:
        super().__init__()

    def _output(self, inference: bool) -> ndarray:
        # 음수는 0으로 잘라내고, 양수는 그대로 둔다.
        return np.clip(self.input_, 0, None)

    def _input_grad(self, output_grad: ndarray) -> ndarray:
        # 출력이 0보다 큰 위치만 gradient를 통과시킨다.
        mask = self.output >= 0
        return output_grad * mask
