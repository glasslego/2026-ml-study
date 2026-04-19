"""numpy만으로 구현한 2-layer MLP (forward + backward).

이 모듈은 PyTorch 같은 자동미분 라이브러리 없이, chain rule을 한 줄씩
직접 적어 내려가면 어떤 식이 나오는지 보여주기 위한 학습용 구현이다.

모델 구조
---------
입력 x → Linear(W1, b1) → z1 → ReLU → a1 → Linear(W2, b2) → y_pred
                                                              │
                                           y_true ──→ MSE Loss

차원 (기본 테스트 설정)
----------------------
- x       : (batch=4, in_dim=3)
- W1, b1  : (3, 5), (5,)   → z1, a1 : (4, 5)
- W2, b2  : (5, 2), (2,)   → y_pred : (4, 2)
- y_true  : (4, 2),  Loss는 스칼라.

역전파 수식 (MSE, N = batch × out_dim)
--------------------------------------
    L                = (1/N) Σ (ŷ - y)^2
    dL/dŷ            = (2/N) (ŷ - y)
    dL/dW2           = a1ᵀ · dL/dŷ
    dL/db2           = Σ_batch dL/dŷ
    dL/da1           = dL/dŷ · W2ᵀ
    dL/dz1           = dL/da1 ⊙ 1[z1 > 0]      (ReLU 미분)
    dL/dW1           = xᵀ · dL/dz1
    dL/db1           = Σ_batch dL/dz1

핵심 직관
---------
- "외적 패턴" (``xᵀ @ δ``) : 선형층의 가중치 gradient는 항상 입력과 상류
  gradient의 외적으로 계산된다.
- "fan-out 합산" (``δ.sum(axis=0)``) : bias는 모든 배치 샘플에 동일한 값이
  broadcast되므로, 역전파 시 모두 더해야 한 번의 업데이트량이 된다.
- "Wᵀ 등장" (``δ @ Wᵀ``) : 다음 층으로 전파되는 gradient는 가중치의
  transpose를 곱한 값이다. forward에서 ``x @ W``였던 것이 반대 방향으로
  돌아오는 구조.
"""

from __future__ import annotations

from typing import Dict

import numpy as np

# -----------------------------------------------------------------------------
# 손실 함수
# -----------------------------------------------------------------------------


def mse_loss(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    """Mean Squared Error를 스칼라로 반환.

    식: L = (1 / N) * Σ_i (ŷ_i - y_i)^2
    여기서 N = y_pred.size = batch × out_dim. 즉 모든 원소에 대한 평균이다.

    Args:
        y_pred: 모델 예측값. shape (batch, out_dim).
        y_true: 정답 타깃. shape (batch, out_dim).

    Returns:
        배치/차원 전체 평균된 손실값 (파이썬 float).
    """
    # (ŷ - y) ** 2 의 모든 원소를 평균 → 스칼라
    # .mean()을 쓰면 "N = batch × out_dim"으로 자동 계산되어,
    # backward에서 (2/N) 상수를 일관되게 맞출 수 있다.
    return float(((y_pred - y_true) ** 2).mean())


# -----------------------------------------------------------------------------
# 2-layer MLP
# -----------------------------------------------------------------------------


class TwoLayerMLP:
    """numpy 손코딩 2-layer MLP.

    초기 가중치를 dict로 받아서 ``self.weights``에 보관하고,
    ``forward()``에서 중간 활성값을 ``self._cache``에 저장해둔다.
    ``backward()``에서 이 캐시를 꺼내 chain rule을 역순으로 계산한다.

    설계 메모
    ---------
    - 가중치를 dict로 관리하는 이유: PyTorch 모델과 키를 맞춰두면 테스트에서
      양쪽 gradient를 이름으로 짝짓기가 쉽다 ("W1", "b1", "W2", "b2").
    - float64로 고정: backprop 검증은 수치 정밀도가 중요해서 float32보다
      float64가 안전하다 (``atol=1e-10`` 수준까지 비교 가능).
    - "한 step 학습시키기"용 API는 일부러 넣지 않았다. 학습은 테스트에서
      사용자가 직접 ``model.weights[k] -= lr * grads[k]`` 로 간단히 한 번
      굴려보도록 두어서, 학습 루프 내부 흐름을 명시적으로 만든다.
    """

    def __init__(self, weights: Dict[str, np.ndarray]) -> None:
        """가중치 dict를 받아 모델을 초기화.

        Args:
            weights: 다음 네 키를 반드시 포함하는 dict.
                - ``W1``: (in_dim, hidden_dim) 첫 번째 Linear의 가중치
                - ``b1``: (hidden_dim,)         첫 번째 Linear의 bias
                - ``W2``: (hidden_dim, out_dim) 두 번째 Linear의 가중치
                - ``b2``: (out_dim,)            두 번째 Linear의 bias

        Note:
            각 배열은 float64로 변환되어 복사된다. 외부에서 넘긴 numpy
            배열을 그대로 참조하면 테스트 간 상태가 공유되어 디버깅이
            어렵기 때문이다.
        """
        # .astype(np.float64)는 새 배열을 만들며 dtype을 맞춘다.
        # float64를 고정하는 이유는 위 "설계 메모" 참조.
        self.weights: Dict[str, np.ndarray] = {
            "W1": weights["W1"].astype(np.float64),
            "b1": weights["b1"].astype(np.float64),
            "W2": weights["W2"].astype(np.float64),
            "b2": weights["b2"].astype(np.float64),
        }

        # forward에서 계산한 중간값을 backward가 쓸 수 있게 저장해두는 공간.
        # 맨 처음에는 비어 있고, forward()가 호출되어야 채워진다.
        self._cache: Dict[str, np.ndarray] = {}

    # -------------------------------------------------------------------------
    # Forward
    # -------------------------------------------------------------------------
    def forward(self, x: np.ndarray) -> np.ndarray:
        """순전파를 계산하고 중간값을 캐싱.

        그림으로 보면:
            x ──@ W1──→ z1 ──ReLU──→ a1 ──@ W2──→ y_pred

        Args:
            x: 입력 배치. shape (batch, in_dim).

        Returns:
            y_pred. shape (batch, out_dim).

        Raises:
            없음 — shape가 틀리면 numpy가 자체 에러를 낸다.
        """
        # --- 첫 번째 Linear 층: z1 = x W1 + b1 ---
        # x:        (batch, in_dim)
        # W1:       (in_dim, hidden_dim)
        # x @ W1:   (batch, hidden_dim)
        # + b1:     bias는 (hidden_dim,) 이므로 broadcast되어 모든 배치 행에
        #           동일한 값이 더해진다. 이 broadcast가 backward에서
        #           "bias gradient = batch 합산" 규칙의 원인이 된다.
        z1 = x @ self.weights["W1"] + self.weights["b1"]

        # --- ReLU 활성화: a1 = max(0, z1) ---
        # np.maximum(0, z1): z1이 음수이면 0, 양수이면 그대로 전달.
        # ReLU의 미분은 "z1 > 0" 인 위치에서 1, 아니면 0.
        a1 = np.maximum(0.0, z1)

        # --- 두 번째 Linear 층: y_pred = a1 W2 + b2 ---
        y_pred = a1 @ self.weights["W2"] + self.weights["b2"]

        # --- 캐시 저장 ---
        # backward에서 다시 필요한 중간값만 저장한다:
        #   x, z1, a1   : 각 층의 입력 (가중치 gradient 계산에 필요)
        #   y_pred      : 손실 gradient 계산의 시작점
        # 메모리 낭비 같지만, 이게 PyTorch autograd가 내부적으로 하는 일과
        # 본질적으로 동일하다. PyTorch는 텐서 단위로 "computation graph"에
        # 이 중간값들을 붙여두고, .backward() 호출 시 되돌아가며 사용한다.
        self._cache = {
            "x": x,
            "z1": z1,
            "a1": a1,
            "y_pred": y_pred,
        }

        return y_pred

    # -------------------------------------------------------------------------
    # Backward
    # -------------------------------------------------------------------------
    def backward(self, y_true: np.ndarray) -> Dict[str, np.ndarray]:
        """역전파로 각 파라미터의 gradient 계산.

        ``forward()``가 먼저 호출되어 ``self._cache``에 중간값이 있어야 한다.

        Args:
            y_true: 정답 타깃. shape (batch, out_dim).

        Returns:
            네 파라미터의 gradient를 담은 dict:
              - ``W1``: shape (in_dim, hidden_dim)
              - ``b1``: shape (hidden_dim,)
              - ``W2``: shape (hidden_dim, out_dim)
              - ``b2``: shape (out_dim,)

        Raises:
            KeyError: forward()를 먼저 호출하지 않은 경우.
        """
        # --- 캐시에서 forward 중간값 꺼내기 ---
        x = self._cache["x"]
        z1 = self._cache["z1"]
        a1 = self._cache["a1"]
        y_pred = self._cache["y_pred"]

        # N = 평균에 쓰인 원소 수. MSE가 .mean()이므로 batch × out_dim.
        # 이 값이 (2/N) 상수에 반영되어야 PyTorch와 수치가 정확히 일치한다.
        N = y_pred.size

        # ---------------------------------------------------------------------
        # Step 1. 손실 L에 대한 y_pred의 gradient
        #   L = (1/N) Σ (ŷ - y)^2
        #   dL/dŷ = (2/N) (ŷ - y)
        # shape: (batch, out_dim)
        # ---------------------------------------------------------------------
        dy_pred = (2.0 / N) * (y_pred - y_true)

        # ---------------------------------------------------------------------
        # Step 2. 두 번째 Linear 층의 파라미터 gradient
        #
        # y_pred = a1 @ W2 + b2
        #   → dL/dW2 = a1ᵀ @ dL/dŷ   (외적 패턴)
        #   → dL/db2 = Σ_batch dL/dŷ (bias는 broadcast → fan-out 합산)
        #
        # shape 확인:
        #   a1.T        : (hidden_dim, batch)
        #   dy_pred     : (batch,      out_dim)
        #   a1.T @ dy   : (hidden_dim, out_dim)   ← W2와 동일 shape
        # ---------------------------------------------------------------------
        dW2 = a1.T @ dy_pred
        # axis=0 : batch 축을 따라 합산. 결과 shape (out_dim,) = b2와 동일.
        db2 = dy_pred.sum(axis=0)

        # ---------------------------------------------------------------------
        # Step 3. 두 번째 Linear 층을 지나 a1까지 gradient 전파
        #
        # y_pred = a1 @ W2 + b2 에서 a1에 대한 미분:
        #   dL/da1 = dL/dŷ @ W2ᵀ   ("Wᵀ 등장" — 순전파의 반대 방향으로 전파)
        #
        # shape:
        #   dy_pred   : (batch, out_dim)
        #   W2.T      : (out_dim, hidden_dim)
        #   결과      : (batch, hidden_dim)   ← a1과 동일 shape
        # ---------------------------------------------------------------------
        da1 = dy_pred @ self.weights["W2"].T

        # ---------------------------------------------------------------------
        # Step 4. ReLU를 역방향으로 통과
        #
        # a1 = ReLU(z1) 에서 z1에 대한 미분:
        #   da1/dz1 = 1  (z1 > 0 인 위치)
        #           = 0  (그 외)
        #   → dL/dz1 = dL/da1 ⊙ 1[z1 > 0]
        #
        # 구현은 "mask 곱" 한 줄. (z1 > 0)은 boolean array인데 float 곱셈 시
        # True=1.0, False=0.0으로 자동 변환된다.
        # ---------------------------------------------------------------------
        dz1 = da1 * (z1 > 0)

        # ---------------------------------------------------------------------
        # Step 5. 첫 번째 Linear 층의 파라미터 gradient
        #   dL/dW1 = xᵀ @ dL/dz1
        #   dL/db1 = Σ_batch dL/dz1
        #
        # Step 2와 완전히 같은 패턴이 반복된다 — "100층이 있어도 이 8줄이
        # 반복될 뿐"이라는 말의 의미가 여기서 드러난다.
        # ---------------------------------------------------------------------
        dW1 = x.T @ dz1
        db1 = dz1.sum(axis=0)

        # 최종 gradient dict를 돌려준다. 키를 self.weights와 맞춰서 바깥에서
        # zip처럼 업데이트할 수 있게 해둔다:
        #     for k in model.weights:
        #         model.weights[k] -= lr * grads[k]
        return {
            "W1": dW1,
            "b1": db1,
            "W2": dW2,
            "b2": db2,
        }
