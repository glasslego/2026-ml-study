"""완전연결층을 이루는 기본 연산들."""

import numpy as np
from numpy import ndarray

from .base import ParamOperation


class WeightMultiply(ParamOperation):
    """입력과 가중치 행렬을 곱하는 연산."""

    def __init__(self, W: ndarray):
        super().__init__(W)

    def _output(self, inference: bool) -> ndarray:
        # (batch, in_features) x (in_features, out_features)
        # -> (batch, out_features)
        return np.matmul(self.input_, self.param)

    def _input_grad(self, output_grad: ndarray) -> ndarray:
        # y = XW 일 때 dL/dX = dL/dY * W^T
        return np.matmul(output_grad, self.param.transpose(1, 0))

    def _param_grad(self, output_grad: ndarray) -> ndarray:
        # y = XW 일 때 dL/dW = X^T * dL/dY
        return np.matmul(self.input_.transpose(1, 0), output_grad)


class BiasAdd(ParamOperation):
    """편향 벡터를 모든 배치 샘플에 더하는 연산."""

    def __init__(self, B: ndarray):
        super().__init__(B)

    def _output(self, inference: bool) -> ndarray:
        # self.param의 shape은 (1, out_features)이며 broadcasting으로 각 행에 더해진다.
        return self.input_ + self.param

    def _input_grad(self, output_grad: ndarray) -> ndarray:
        # y = x + b 에서 x에 대한 도함수는 1이므로 gradient가 그대로 전달된다.
        return np.ones_like(self.input_) * output_grad

    def _param_grad(self, output_grad: ndarray) -> ndarray:
        # 하나의 bias가 배치 전체에서 재사용되므로 batch 축으로 gradient를 모두 합친다.
        output_grad_reshape = np.sum(output_grad, axis=0).reshape(1, -1)
        param_grad = np.ones_like(self.param)
        return param_grad * output_grad_reshape
