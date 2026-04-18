"""Layer 묶음을 네트워크처럼 다루는 모듈."""

from typing import List

from numpy import ndarray

from .layers import Layer
from .losses import Loss, MeanSquaredError


class LayerBlock(object):
    """여러 layer를 순서대로 실행하는 컨테이너."""

    def __init__(self, layers: List[Layer]):
        super().__init__()
        self.layers = layers

    def forward(self, X_batch: ndarray, inference=False) -> ndarray:
        """배치 데이터를 앞에서 뒤로 통과시킨다."""

        X_out = X_batch
        for layer in self.layers:
            X_out = layer.forward(X_out, inference)

        return X_out

    def backward(self, loss_grad: ndarray) -> ndarray:
        """loss에서 온 gradient를 뒤에서 앞으로 전달한다."""

        grad = loss_grad
        for layer in reversed(self.layers):
            grad = layer.backward(grad)

        return grad

    def params(self):
        """모든 layer의 파라미터를 순서대로 꺼내는 generator."""

        for layer in self.layers:
            yield from layer.params

    def param_grads(self):
        """모든 layer의 파라미터 gradient를 순서대로 꺼내는 generator."""

        for layer in self.layers:
            yield from layer.param_grads

    def __iter__(self):
        return iter(self.layers)

    def __repr__(self):
        layer_strs = [str(layer) for layer in self.layers]
        return f"{self.__class__.__name__}(\n  " + ",\n  ".join(layer_strs) + ")"


class NeuralNetwork(LayerBlock):
    """손실함수까지 연결한 간단한 신경망."""

    def __init__(
        self,
        layers: List[Layer],
        loss: Loss = MeanSquaredError,
        seed: int = 1,
    ):
        super().__init__(layers)
        self.loss = loss
        self.seed = seed
        if seed:
            for layer in self.layers:
                setattr(layer, "seed", self.seed)

    def forward_loss(
        self,
        X_batch: ndarray,
        y_batch: ndarray,
        inference: bool = False,
    ) -> float:
        """순전파와 loss 계산만 수행한다."""

        prediction = self.forward(X_batch, inference)
        return self.loss.forward(prediction, y_batch)

    def train_batch(
        self,
        X_batch: ndarray,
        y_batch: ndarray,
        inference: bool = False,
    ) -> float:
        """한 배치에 대해 순전파, loss 계산, 역전파까지 수행한다."""

        prediction = self.forward(X_batch, inference)

        batch_loss = self.loss.forward(prediction, y_batch)
        loss_grad = self.loss.backward()

        # loss의 gradient가 네트워크 역전파의 시작점이 된다.
        self.backward(loss_grad)

        return batch_loss
