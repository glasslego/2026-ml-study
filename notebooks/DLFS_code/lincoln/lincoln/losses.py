"""손실함수와 그 gradient 계산."""

import numpy as np
from numpy import ndarray

from lincoln.utils.np_utils import (
    assert_same_shape,
    exp_ratios,
    softmax,
    normalize,
    unnormalize,
)


class Loss(object):
    """모든 손실함수가 공유하는 공통 인터페이스."""

    def __init__(self):
        pass

    def forward(self, prediction: ndarray, target: ndarray) -> float:
        """예측값과 정답을 받아 손실을 계산한다."""

        assert_same_shape(prediction, target)

        self.prediction = prediction
        self.target = target

        self.output = self._output()

        return self.output

    def backward(self) -> ndarray:
        """손실을 예측값 기준 gradient로 바꾼다."""

        self.input_grad = self._input_grad()

        assert_same_shape(self.prediction, self.input_grad)

        return self.input_grad

    def _output(self) -> float:
        raise NotImplementedError()

    def _input_grad(self) -> ndarray:
        raise NotImplementedError()


class MeanSquaredError(Loss):
    """평균제곱오차(MSE) 손실."""

    def __init__(self, normalize: bool = False) -> None:
        super().__init__()
        self.normalize = normalize

    def _output(self) -> float:
        if self.normalize:
            self.prediction = self.prediction / self.prediction.sum(axis=1, keepdims=True)

        loss = (
            np.sum(np.power(self.prediction - self.target, 2))
            / self.prediction.shape[0]
        )

        return loss

    def _input_grad(self) -> ndarray:
        return 2.0 * (self.prediction - self.target) / self.prediction.shape[0]


class SoftmaxCrossEntropy(Loss):
    """softmax와 cross entropy를 결합한 손실."""

    def __init__(self, eps: float = 1e-9) -> None:
        super().__init__()
        self.eps = eps
        self.single_class = False

    def _output(self) -> float:
        # 원 코드의 흐름을 보존한다.
        if self.target.shape[1] == 0:
            self.single_class = True

        if self.single_class:
            self.prediction, self.target = normalize(self.prediction), normalize(
                self.target
            )

        softmax_preds = softmax(self.prediction, axis=1)

        # log(0)을 피하기 위해 확률값을 아주 조금 안쪽으로 잘라 준다.
        self.softmax_preds = np.clip(softmax_preds, self.eps, 1 - self.eps)

        softmax_cross_entropy_loss = (
            -1.0 * self.target * np.log(self.softmax_preds)
            - (1.0 - self.target) * np.log(1 - self.softmax_preds)
        )

        return np.sum(softmax_cross_entropy_loss) / self.prediction.shape[0]

    def _input_grad(self) -> ndarray:
        # softmax + cross entropy 조합은 미분 결과가
        # (softmax 확률 - 정답)으로 단순화되는 것이 핵심이다.
        if self.single_class:
            return unnormalize(self.softmax_preds - self.target)
        return (self.softmax_preds - self.target) / self.prediction.shape[0]


class SoftmaxCrossEntropyComplex(SoftmaxCrossEntropy):
    """Jacobian을 직접 전개하는 학습용 구현."""

    def __init__(self, eta: float = 1e-9, single_output: bool = False) -> None:
        super().__init__()
        self.single_output = single_output

    def _input_grad(self) -> ndarray:
        prob_grads = []
        batch_size = self.softmax_preds.shape[0]
        num_features = self.softmax_preds.shape[1]
        for n in range(batch_size):
            exp_ratio = exp_ratios(self.prediction[n] - np.max(self.prediction[n]))
            jacobian = np.zeros((num_features, num_features))
            for f1 in range(num_features):
                for f2 in range(num_features):
                    if f1 == f2:
                        jacobian[f1][f2] = (
                            self.softmax_preds[n][f1] - self.target[n][f1]
                        )
                    else:
                        jacobian[f1][f2] = (
                            -(self.target[n][f2] - 1) * exp_ratio[f1][f2]
                            + self.target[n][f2]
                            + self.softmax_preds[n][f1]
                            - 1
                        )
            prob_grads.append(jacobian.sum(axis=1))

        if self.single_class:
            return unnormalize(np.stack(prob_grads))
        return np.stack(prob_grads)
