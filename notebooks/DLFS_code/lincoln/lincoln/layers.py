"""Operation을 묶어 하나의 layer처럼 다루는 모듈."""

from typing import List

import numpy as np
from numpy import ndarray

from .activations import Linear
from .base import Operation, ParamOperation
from .conv import Conv2D_Op
from .dense import WeightMultiply, BiasAdd
from .dropout import Dropout
from .reshape import Flatten
from lincoln.utils.np_utils import assert_same_shape


class Layer(object):
    """여러 Operation을 순서대로 실행하는 추상 layer."""

    def __init__(self, neurons: int) -> None:
        self.neurons = neurons
        self.first = True
        self.params: List[ndarray] = []
        self.param_grads: List[ndarray] = []
        self.operations: List[Operation] = []

    def _setup_layer(self, input_: ndarray) -> None:
        """첫 입력을 보고 파라미터/연산 목록을 준비한다."""

        pass

    def forward(self, input_: ndarray, inference=False) -> ndarray:
        """Layer 안의 모든 Operation을 앞에서 뒤로 실행한다."""

        if self.first:
            # Dense layer는 입력 feature 수를 실제 데이터를 보고 알아내야 하므로
            # 첫 forward 시점에 파라미터를 초기화한다.
            self._setup_layer(input_)
            self.first = False

        self.input_ = input_

        for operation in self.operations:
            input_ = operation.forward(input_, inference)

        self.output = input_

        return self.output

    def backward(self, output_grad: ndarray) -> ndarray:
        """Layer 안의 Operation을 역순으로 돌며 gradient를 전달한다."""

        assert_same_shape(self.output, output_grad)

        for operation in self.operations[::-1]:
            output_grad = operation.backward(output_grad)

        input_grad = output_grad

        assert_same_shape(self.input_, input_grad)

        self._param_grads()

        return input_grad

    def _param_grads(self) -> None:
        """현재 layer 안의 학습 가능한 연산들에서 gradient를 수집한다."""

        self.param_grads = []
        for operation in self.operations:
            if issubclass(operation.__class__, ParamOperation):
                self.param_grads.append(operation.param_grad)

    def _params(self) -> None:
        """현재 layer 안의 학습 가능한 파라미터를 수집한다."""

        self.params = []
        for operation in self.operations:
            if issubclass(operation.__class__, ParamOperation):
                self.params.append(operation.param)


class Dense(Layer):
    """완전연결층을 구성하는 layer."""

    def __init__(
        self,
        neurons: int,
        activation: Operation = Linear(),
        conv_in: bool = False,
        dropout: float = 1.0,
        weight_init: str = "standard",
    ) -> None:
        super().__init__(neurons)
        self.activation = activation
        self.conv_in = conv_in
        self.dropout = dropout
        self.weight_init = weight_init

    def _setup_layer(self, input_: ndarray) -> None:
        np.random.seed(self.seed)
        num_in = input_.shape[1]

        if self.weight_init == "glorot":
            # Glorot 초기화는 입력/출력 크기에 맞춰 분산을 줄여
            # gradient가 너무 커지거나 작아지는 문제를 완화한다.
            scale = 2 / (num_in + self.neurons)
        else:
            scale = 1.0

        self.params = []
        self.params.append(
            np.random.normal(loc=0, scale=scale, size=(num_in, self.neurons))
        )

        self.params.append(
            np.random.normal(loc=0, scale=scale, size=(1, self.neurons))
        )

        self.operations = [
            WeightMultiply(self.params[0]),
            BiasAdd(self.params[1]),
            self.activation,
        ]

        if self.dropout < 1.0:
            self.operations.append(Dropout(self.dropout))

        return None


class Conv2D(Layer):
    """합성곱 연산, 활성화, flatten/dropout을 묶는 layer."""

    def __init__(
        self,
        out_channels: int,
        param_size: int,
        dropout: int = 1.0,
        weight_init: str = "normal",
        activation: Operation = Linear(),
        flatten: bool = False,
    ) -> None:
        super().__init__(out_channels)
        self.param_size = param_size
        self.activation = activation
        self.flatten = flatten
        self.dropout = dropout
        self.weight_init = weight_init
        self.out_channels = out_channels

    def _setup_layer(self, input_: ndarray) -> ndarray:
        self.params = []
        in_channels = input_.shape[1]

        if self.weight_init == "glorot":
            scale = 2 / (in_channels + self.out_channels)
        else:
            scale = 1.0

        conv_param = np.random.normal(
            loc=0,
            scale=scale,
            size=(
                input_.shape[1],  # 입력 채널 수
                self.out_channels,
                self.param_size,
                self.param_size,
            ),
        )

        self.params.append(conv_param)

        self.operations = []
        self.operations.append(Conv2D_Op(conv_param))
        self.operations.append(self.activation)

        if self.flatten:
            # 합성곱 출력 뒤에 Dense layer를 연결할 때 4D 텐서를 2D로 편다.
            self.operations.append(Flatten())

        if self.dropout < 1.0:
            self.operations.append(Dropout(self.dropout))

        return None
