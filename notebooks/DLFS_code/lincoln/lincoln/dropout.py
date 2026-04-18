"""드롭아웃 연산 구현."""

import numpy as np
from numpy import ndarray

from .base import Operation


class Dropout(Operation):
    """일부 뉴런 출력을 무작위로 꺼서 과적합을 줄이는 연산."""

    def __init__(self, keep_prob: float = 0.8):
        super().__init__()
        self.keep_prob = keep_prob

    def _output(self, inference: bool) -> ndarray:
        if inference:
            # 이 구현은 추론 시에 keep_prob를 곱해 학습 시 기대값과 맞춘다.
            return self.input_ * self.keep_prob

        # 학습 시에는 0/1 마스크를 샘플링해 일부 뉴런만 살아남게 한다.
        self.mask = np.random.binomial(1, self.keep_prob, size=self.input_.shape)
        return self.input_ * self.mask

    def _input_grad(self, output_grad: ndarray) -> ndarray:
        # 순전파 때 꺼졌던 뉴런은 역전파 gradient도 0이어야 하므로 같은 mask를 재사용한다.
        return output_grad * self.mask
