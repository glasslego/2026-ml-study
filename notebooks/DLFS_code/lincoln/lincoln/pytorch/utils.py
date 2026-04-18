"""PyTorch 버전에서 쓰는 작은 유틸리티 함수."""

from typing import Tuple

import torch
from torch import Tensor


def permute_data(X: Tensor, y: Tensor, seed=1) -> Tuple[Tensor]:
    """특징과 정답 텐서를 같은 순서로 함께 섞는다."""

    perm = torch.randperm(X.shape[0])
    return X[perm], y[perm]


def assert_dim(t: Tensor, dim: int):
    """텐서 차원 수가 기대값과 같은지 확인한다."""

    assert len(t.shape) == dim, """
        Tensor expected to have dimension {0}, instead has dimension {1}
        """.format(dim, len(t.shape))
    return None
