"""numpy로 손코딩한 backprop을 PyTorch autograd와 비교 검증하는 테스트.

왜 PyTorch를 oracle로 쓰는가?
-----------------------------
- PyTorch autograd는 수천만 사용자가 매일 쓰는, 이미 충분히 검증된 구현.
- "계산 그래프를 역순으로 따라가며 chain rule을 적용한다"는 알고리즘은
  구현이 다르더라도 같은 수식을 따른다면 float64 기준 1e-10 수준에서 일치
  해야 한다.
- 따라서 numpy 구현이 PyTorch와 이만큼 가깝다면, 우리가 적은 backward
  수식이 맞다는 강력한 증거가 된다.

테스트 설계
------------
1. ``toy_batch`` : 같은 seed로 만든 작은 (batch=4, in_dim=3) 입력.
2. ``initial_weights`` : 같은 seed로 만든 W1/b1/W2/b2 초기값. numpy 모델과
   PyTorch 모델 양쪽에 동일하게 주입한다 — 초기값이 달라지면 비교 의미가
   사라지므로 가장 중요한 전제.
3. ``torch_oracle()`` : 같은 입력/가중치로 PyTorch 모델을 만들어 loss와
   gradient를 돌려주는 함수. 이것이 "정답 제공자(oracle)" 역할.
4. 각 파라미터마다 독립 테스트 (``test_grad_W2_...``, ``test_grad_b2_...``,
   ``test_grad_W1_...``, ``test_grad_b1_...``)를 두어서, 어느 한 항목만
   틀렸을 때 정확히 어디가 틀렸는지 진단할 수 있게 한다.
5. ``test_forward_loss_matches_pytorch`` : forward 경로만 먼저 검증. 이게
   틀리면 backward를 아무리 봐야 소용없으므로 순서가 중요하다.
6. ``test_one_training_step_reduces_loss`` : gradient descent를 한 step
   돌렸을 때 loss가 내려가는지 확인하는 smoke test. backward가 맞아도
   부호가 뒤집혀 있으면 (예: ``+= lr * grad``) 이 테스트만 유일하게
   잡아낸다.

atol 값 고르기
---------------
- float64 machine epsilon ≈ 2.2e-16 이므로, 수식이 정확히 같다면 차이는
  1e-14 ~ 1e-16 수준에 머문다.
- ``atol=1e-10`` 은 "구현이 본질적으로 동일하다"고 말할 수 있는 관용
  한계. 이보다 크게 벌어지면 어딘가 수식이 다르거나 dtype이 섞인 것.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from mlp import TwoLayerMLP, mse_loss


# -----------------------------------------------------------------------------
# 공용 픽스처 — 모든 테스트가 같은 입력/가중치를 쓰도록 고정
# -----------------------------------------------------------------------------


@pytest.fixture
def toy_batch() -> tuple[np.ndarray, np.ndarray]:
    """재현 가능한 작은 배치. (batch=4, in_dim=3, out_dim=2).

    seed=42로 고정해 매 실행마다 동일한 x, y를 만든다. 재현성이 없으면
    어제는 통과하던 테스트가 오늘 깨지는 "유령 버그"가 생기므로,
    TDD에서는 항상 seed를 명시하는 것이 원칙.

    Returns:
        (x, y) 튜플.
        - x : (4, 3) float64 입력.
        - y : (4, 2) float64 타깃.
    """
    rng = np.random.default_rng(seed=42)
    x = rng.standard_normal(size=(4, 3)).astype(np.float64)
    y = rng.standard_normal(size=(4, 2)).astype(np.float64)
    return x, y


@pytest.fixture
def initial_weights() -> dict[str, np.ndarray]:
    """numpy/PyTorch 양쪽에 주입할 동일한 초기 가중치.

    ``* 0.5`` 스케일링은 작은 네트워크에서 ReLU가 절반 가량 살아남도록
    값의 범위를 조금 좁혀두기 위함 (너무 크면 모든 뉴런이 포화되어
    ReLU의 dead-neuron 효과가 테스트를 덮어버림).
    """
    rng = np.random.default_rng(seed=123)
    return {
        "W1": rng.standard_normal(size=(3, 5)).astype(np.float64) * 0.5,
        "b1": np.zeros(5, dtype=np.float64),
        "W2": rng.standard_normal(size=(5, 2)).astype(np.float64) * 0.5,
        "b2": np.zeros(2, dtype=np.float64),
    }


# -----------------------------------------------------------------------------
# Oracle — PyTorch autograd로 "정답" loss/gradient를 계산
# -----------------------------------------------------------------------------


def torch_oracle(
    weights: dict[str, np.ndarray],
    x: np.ndarray,
    y: np.ndarray,
) -> tuple[float, dict[str, np.ndarray]]:
    """동일한 입력/가중치로 PyTorch 모델을 돌려 loss와 gradient를 반환.

    numpy 모델과 1:1로 같은 구조·수식·순서를 따라야 비교의 의미가 있다:
        z1 = x @ W1 + b1
        a1 = relu(z1)
        y_pred = a1 @ W2 + b2
        loss = mean((y_pred - y)**2)

    ``requires_grad=True``로 표시된 텐서에만 gradient가 기록되고,
    ``loss.backward()`` 호출이 전체 계산 그래프를 역방향 훑으며
    모든 ``.grad``를 채워준다.

    Args:
        weights: numpy 모델과 동일한 초기 가중치 dict.
        x, y:    동일한 입력/타깃 배치.

    Returns:
        (loss_scalar, grad_dict) 튜플.
          - loss_scalar : python float.
          - grad_dict   : {"W1", "b1", "W2", "b2"} → numpy 배열.
    """
    # torch.tensor(..., requires_grad=True)는 "이 텐서에 대한 gradient를
    # 자동으로 추적해달라"는 선언. numpy 가중치를 그대로 넘겨주면 PyTorch
    # 내부에서 float64 텐서로 복사된다.
    params = {k: torch.tensor(v, requires_grad=True) for k, v in weights.items()}

    # x, y는 "입력"이라 gradient가 필요 없으므로 requires_grad 생략.
    x_t = torch.tensor(x)
    y_t = torch.tensor(y)

    # --- numpy 모델과 동일한 forward 계산 ---
    # 연산자 @는 torch에서도 matmul. broadcast 규칙도 동일.
    z1 = x_t @ params["W1"] + params["b1"]
    a1 = torch.relu(z1)
    y_pred = a1 @ params["W2"] + params["b2"]
    loss = ((y_pred - y_t) ** 2).mean()

    # 이 한 줄이 전체 역전파를 실행한다. 내부적으로 PyTorch는 forward
    # 동안 구축해둔 "Function graph"를 역방향으로 훑으며, 각 연산에
    # 등록된 backward 규칙을 적용한다. 결과가 param.grad에 누적됨.
    loss.backward()

    # .detach().numpy()는 gradient 추적 그래프에서 분리한 뒤 numpy 배열로
    # 복사. 테스트에서 np.testing.assert_allclose로 비교할 것이므로 numpy
    # 배열 형태가 편하다.
    grads = {k: p.grad.detach().numpy() for k, p in params.items()}
    return loss.item(), grads


# -----------------------------------------------------------------------------
# 테스트 케이스들
# -----------------------------------------------------------------------------


def test_forward_loss_matches_pytorch(toy_batch, initial_weights):
    """forward 경로가 맞는지 먼저 검증.

    backward가 틀렸더라도 forward가 틀렸으면 애초에 비교할 값이 없다.
    따라서 "숫자 흐름의 출발점"인 loss부터 점검해야 디버깅 범위가 좁아진다.
    """
    x, y = toy_batch
    model = TwoLayerMLP(weights=initial_weights)

    y_pred = model.forward(x)
    loss = mse_loss(y_pred, y)

    expected_loss, _ = torch_oracle(initial_weights, x, y)
    # isclose는 "상대 + 절대 오차"를 함께 보는 비교. atol=1e-10은 float64
    # 기준으로 "실질적으로 동일"에 해당하는 엄격한 기준.
    assert np.isclose(loss, expected_loss, atol=1e-10), (
        f"forward loss mismatch: got {loss}, expected {expected_loss}"
    )


def test_grad_W2_matches_pytorch(toy_batch, initial_weights):
    """두 번째 Linear 층의 가중치 gradient 검증.

    수식: dL/dW2 = a1ᵀ @ dL/dŷ
    이 값이 틀렸다면 보통 ``a1.T`` 대신 ``a1``을 썼거나 matmul 순서가
    뒤집힌 경우가 많다.
    """
    x, y = toy_batch
    model = TwoLayerMLP(weights=initial_weights)
    model.forward(x)
    grads = model.backward(y)

    _, expected = torch_oracle(initial_weights, x, y)
    np.testing.assert_allclose(grads["W2"], expected["W2"], atol=1e-10)


def test_grad_b2_matches_pytorch(toy_batch, initial_weights):
    """두 번째 Linear 층의 bias gradient 검증.

    수식: dL/db2 = Σ_batch dL/dŷ
    이 값이 틀렸다면 보통 ``.sum(axis=0)``을 잊어서 shape가 (batch, out_dim)
    으로 남아 있는 경우.
    """
    x, y = toy_batch
    model = TwoLayerMLP(weights=initial_weights)
    model.forward(x)
    grads = model.backward(y)

    _, expected = torch_oracle(initial_weights, x, y)
    np.testing.assert_allclose(grads["b2"], expected["b2"], atol=1e-10)


def test_grad_W1_matches_pytorch(toy_batch, initial_weights):
    """첫 번째 Linear 층의 가중치 gradient 검증.

    수식: dL/dW1 = xᵀ @ dL/dz1
    여기서 dL/dz1 = (dL/dŷ @ W2ᵀ) ⊙ 1[z1 > 0].
    ReLU mask를 빼먹으면 이 테스트가 제일 먼저 깨진다.
    """
    x, y = toy_batch
    model = TwoLayerMLP(weights=initial_weights)
    model.forward(x)
    grads = model.backward(y)

    _, expected = torch_oracle(initial_weights, x, y)
    np.testing.assert_allclose(grads["W1"], expected["W1"], atol=1e-10)


def test_grad_b1_matches_pytorch(toy_batch, initial_weights):
    """첫 번째 Linear 층의 bias gradient 검증.

    수식: dL/db1 = Σ_batch dL/dz1
    W1과 짝을 이루는 항으로, 둘 다 함께 맞거나 함께 틀리는 경향이 있다.
    """
    x, y = toy_batch
    model = TwoLayerMLP(weights=initial_weights)
    model.forward(x)
    grads = model.backward(y)

    _, expected = torch_oracle(initial_weights, x, y)
    np.testing.assert_allclose(grads["b1"], expected["b1"], atol=1e-10)


def test_one_training_step_reduces_loss(toy_batch, initial_weights):
    """gradient descent 1-step 스모크 테스트.

    backward 방향 부호를 뒤집어도(``+= lr * grad``) 다른 테스트는 통과
    할 수 있다 — 수치 자체는 PyTorch와 같으니까. 이 smoke test는
    "부호와 업데이트 규칙이 말이 되는가"를 검증하는 마지막 안전망.
    """
    x, y = toy_batch
    model = TwoLayerMLP(weights=initial_weights)

    loss_before = mse_loss(model.forward(x), y)
    grads = model.backward(y)

    # 충분히 작은 learning rate. 너무 크면 한 step 만에 오히려 loss가
    # 폭주하는 경우가 있어, 이 테스트의 "1-step 감소" 가정이 깨진다.
    lr = 0.01
    for k in model.weights:
        # gradient descent: w ← w - lr * ∂L/∂w
        # 부호가 "-" 인 이유: gradient는 loss가 증가하는 방향이므로,
        # 내려가려면 반대로 이동해야 한다.
        model.weights[k] -= lr * grads[k]

    loss_after = mse_loss(model.forward(x), y)
    assert loss_after < loss_before, (
        f"loss did not decrease: before={loss_before}, after={loss_after}"
    )
