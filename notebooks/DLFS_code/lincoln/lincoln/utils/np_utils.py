"""NumPy 기반 구현 전반에서 재사용하는 유틸리티 함수."""

from typing import Tuple

import numpy as np
from scipy.special import logsumexp


def to_2d(a: np.ndarray, type: str = "col") -> np.ndarray:
    """1차원 벡터를 2차원 행/열 벡터로 바꾼다."""

    assert a.ndim == 1, "Input tensors must be 1 dimensional"

    if type == "col":
        return a.reshape(-1, 1)
    elif type == "row":
        return a.reshape(1, -1)


def normalize(a: np.ndarray):
    """이진 분류 출력을 2클래스 확률 표현으로 확장한다."""

    other = 1 - a
    return np.concatenate([a, other], axis=1)


def unnormalize(a: np.ndarray):
    """2클래스 표현으로 확장된 값을 다시 단일 출력 형태로 줄인다."""

    return a[np.newaxis, 0]


def permute_data(X: np.ndarray, y: np.ndarray):
    """특징과 정답을 같은 순서로 함께 섞는다."""

    perm = np.random.permutation(X.shape[0])
    return X[perm], y[perm]


Batch = Tuple[np.ndarray, np.ndarray]


def generate_batch(
    X: np.ndarray,
    y: np.ndarray,
    start: int = 0,
    batch_size: int = 10,
) -> Batch:
    """지정한 시작점부터 하나의 mini-batch를 잘라 반환한다."""

    # 이 함수는 NumPy 배열을 받으므로 PyTorch의 .dim()이 아니라 .ndim을 써야 한다.
    assert (X.ndim == 2) and (y.ndim == 2), "X and Y must be 2 dimensional"

    if start + batch_size > X.shape[0]:
        batch_size = X.shape[0] - start

    X_batch, y_batch = X[start : start + batch_size], y[start : start + batch_size]

    return X_batch, y_batch


def assert_same_shape(output: np.ndarray, output_grad: np.ndarray):
    """두 텐서가 같은 모양인지 확인한다."""

    assert output.shape == output_grad.shape, """
        Two tensors should have the same shape;
        instead, first Tensor's shape is {0}
        and second Tensor's shape is {1}.
        """.format(tuple(output_grad.shape), tuple(output.shape))
    return None


def assert_dim(t: np.ndarray, dim: int):
    """텐서 차원 수가 예상과 같은지 확인한다."""

    assert t.ndim == dim, """
        Tensor expected to have dimension {0}, instead has dimension {1}
        """.format(dim, len(t.shape))
    return None


def softmax(x, axis=None):
    """수치적으로 안정적인 softmax를 계산한다."""

    return np.exp(x - logsumexp(x, axis=axis, keepdims=True))


def exp_ratios(x: np.ndarray) -> np.ndarray:
    """softmax Jacobian 전개에 쓰는 exp 비율 행렬을 계산한다."""

    exps = np.exp(x)
    denom = np.power(np.sum(exps), 2)
    return np.outer(exps, exps) / denom
