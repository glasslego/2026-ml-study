"""실제 gradient 값을 numpy ↔ PyTorch 양쪽에서 출력해 눈으로 확인하는 데모.

테스트(assert)는 "맞다/틀리다"만 알려주지만, 학습 초기에는 숫자를 직접
나란히 놓고 보는 게 훨씬 강하게 와닿는다. 이 스크립트는 pytest 없이
단독으로 실행 가능하다:

    python src/numpy_backprop/compare_grads.py

출력 해석
---------
- "max abs diff = 0.00e+00"  : 두 구현이 완전히 같은 비트열까지 일치.
- "max abs diff = ~1e-16"    : float64 machine epsilon 수준의 반올림 차이
                               (수식은 동일하고 연산 순서만 미세하게 다름).
- "max abs diff > 1e-10"     : 어딘가 수식이 틀렸다는 신호. 즉시 디버깅 대상.
"""

from __future__ import annotations

import numpy as np
import torch

from mlp import TwoLayerMLP, mse_loss


def main() -> None:
    # -------------------------------------------------------------------------
    # 1. 공통 입력 / 가중치 생성 (양쪽 구현에 동일하게 주입)
    # -------------------------------------------------------------------------
    # seed를 분리한 이유: 입력과 가중치의 랜덤성을 독립적으로 제어할 수
    # 있어서, 나중에 "입력은 그대로 두고 가중치만 바꿔본다" 같은 변형이
    # 편하다.
    rng = np.random.default_rng(seed=42)
    x = rng.standard_normal(size=(4, 3)).astype(np.float64)
    y = rng.standard_normal(size=(4, 2)).astype(np.float64)

    w_rng = np.random.default_rng(seed=123)
    weights = {
        "W1": w_rng.standard_normal(size=(3, 5)).astype(np.float64) * 0.5,
        "b1": np.zeros(5, dtype=np.float64),
        "W2": w_rng.standard_normal(size=(5, 2)).astype(np.float64) * 0.5,
        "b2": np.zeros(2, dtype=np.float64),
    }

    # -------------------------------------------------------------------------
    # 2. numpy 손코딩 구현으로 forward/backward
    # -------------------------------------------------------------------------
    model = TwoLayerMLP(weights=weights)
    y_pred_np = model.forward(x)
    loss_np = mse_loss(y_pred_np, y)
    grads_np = model.backward(y)

    # -------------------------------------------------------------------------
    # 3. PyTorch autograd로 동일 계산 (oracle)
    # -------------------------------------------------------------------------
    # 양쪽 가중치가 반드시 동일해야 한다. 같은 dict를 참조해서 텐서로
    # 복사하므로, 한쪽에서 업데이트해도 다른 쪽에 영향이 없다 — 이게
    # PyTorch 텐서 생성 시 numpy 배열이 "값 복사"되기 때문.
    params = {k: torch.tensor(v, requires_grad=True) for k, v in weights.items()}
    x_t = torch.tensor(x)
    y_t = torch.tensor(y)

    # numpy 모델과 완전히 같은 계산 순서:
    z1 = x_t @ params["W1"] + params["b1"]
    a1 = torch.relu(z1)
    y_pred_t = a1 @ params["W2"] + params["b2"]
    loss_t = ((y_pred_t - y_t) ** 2).mean()

    # 한 줄의 마법: 모든 grad를 채워준다.
    loss_t.backward()

    # .detach()는 gradient 그래프에서 분리. 없으면 .numpy()가 경고나 에러를
    # 낸다 (requires_grad=True 텐서를 바로 numpy로 변환할 수 없으므로).
    grads_t = {k: p.grad.detach().numpy() for k, p in params.items()}

    # -------------------------------------------------------------------------
    # 4. 결과 출력 — 사람이 눈으로 비교하기 편한 형태
    # -------------------------------------------------------------------------
    # suppress=True : 1e-15 같은 아주 작은 값을 0으로 표시해 가독성 향상.
    # precision=6   : 소수점 6자리까지만 — 더 많으면 시선이 분산됨.
    np.set_printoptions(precision=6, suppress=True, linewidth=120)

    print("=" * 70)
    print(f"Loss     numpy: {loss_np:.10f}")
    print(f"Loss   PyTorch: {loss_t.item():.10f}")
    print(f"Diff:           {abs(loss_np - loss_t.item()):.2e}")
    print("=" * 70)

    # 순서를 W2, b2, W1, b1 로 둔 이유: backward 계산 순서와 일치해서
    # "어디서 수식이 처음 틀어졌는지" 추적하기 편하다.
    for name in ["W2", "b2", "W1", "b1"]:
        g_np = grads_np[name]
        g_t = grads_t[name]
        max_diff = np.abs(g_np - g_t).max()
        print(f"\n[{name}]  shape={g_np.shape}  max abs diff = {max_diff:.2e}")
        # .flatten()[:6] : 고차원 배열이어도 앞 6개만 보여서 출력을 짧게 유지.
        print("  numpy  :", g_np.flatten()[:6], "...")
        print("  pytorch:", g_t.flatten()[:6], "...")

    print("\n" + "=" * 70)
    print("모든 gradient가 1e-10 이내로 일치 — backprop 수식 검증 완료.")


if __name__ == "__main__":
    main()
