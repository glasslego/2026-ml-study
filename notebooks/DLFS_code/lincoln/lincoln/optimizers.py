"""가중치 업데이트 규칙 모음."""

import numpy as np


class Optimizer(object):
    """모든 optimizer가 공유하는 공통 기능."""

    def __init__(
        self,
        lr: float = 0.01,
        final_lr: float = 0,
        decay_type: str = None,
    ) -> None:
        self.lr = lr
        self.final_lr = final_lr
        self.decay_type = decay_type
        self.first = True

    def _setup_decay(self) -> None:
        """학습률 decay에 필요한 계수를 미리 계산한다."""

        if not self.decay_type:
            return
        elif self.decay_type == "exponential":
            self.decay_per_epoch = np.power(
                self.final_lr / self.lr, 1.0 / (self.max_epochs - 1)
            )
        elif self.decay_type == "linear":
            self.decay_per_epoch = (self.lr - self.final_lr) / (self.max_epochs - 1)

    def _decay_lr(self) -> None:
        """epoch가 끝날 때 학습률을 조금씩 줄인다."""

        if not self.decay_type:
            return

        if self.decay_type == "exponential":
            self.lr *= self.decay_per_epoch

        elif self.decay_type == "linear":
            self.lr -= self.decay_per_epoch

    def step(self, epoch: int = 0) -> None:
        """네트워크가 가진 모든 파라미터에 업데이트 규칙을 적용한다."""

        for (param, param_grad) in zip(self.net.params(), self.net.param_grads()):
            self._update_rule(param=param, grad=param_grad)

    def _update_rule(self, **kwargs) -> None:
        raise NotImplementedError()


class SGD(Optimizer):
    """가장 단순한 확률적 경사하강법."""

    def __init__(
        self,
        lr: float = 0.01,
        final_lr: float = 0,
        decay_type: str = None,
    ) -> None:
        super().__init__(lr, final_lr, decay_type)

    def _update_rule(self, **kwargs) -> None:
        update = self.lr * kwargs["grad"]
        kwargs["param"] -= update


class SGDMomentum(Optimizer):
    """이전 이동 방향을 일부 기억하는 SGD."""

    def __init__(
        self,
        lr: float = 0.01,
        final_lr: float = 0,
        decay_type: str = None,
        momentum: float = 0.9,
    ) -> None:
        super().__init__(lr, final_lr, decay_type)
        self.momentum = momentum

    def step(self) -> None:
        if self.first:
            # velocity는 optimizer가 내부적으로 들고 있는 "이전 이동 속도" 상태다.
            self.velocities = [np.zeros_like(param) for param in self.net.params()]
            self.first = False

        for (param, param_grad, velocity) in zip(
            self.net.params(),
            self.net.param_grads(),
            self.velocities,
        ):
            self._update_rule(param=param, grad=param_grad, velocity=velocity)

    def _update_rule(self, **kwargs) -> None:
        kwargs["velocity"] *= self.momentum
        kwargs["velocity"] += self.lr * kwargs["grad"]

        kwargs["param"] -= kwargs["velocity"]


class AdaGrad(Optimizer):
    """좌표별 누적 gradient 크기를 반영하는 optimizer."""

    def __init__(
        self,
        lr: float = 0.01,
        final_lr_exp: float = 0,
        final_lr_linear: float = 0,
    ) -> None:
        super().__init__(lr, final_lr_exp, final_lr_linear)
        self.eps = 1e-7

    def step(self) -> None:
        if self.first:
            self.sum_squares = [np.zeros_like(param) for param in self.net.params()]
            self.first = False

        for (param, param_grad, sum_square) in zip(
            self.net.params(),
            self.net.param_grads(),
            self.sum_squares,
        ):
            self._update_rule(param=param, grad=param_grad, sum_square=sum_square)

    def _update_rule(self, **kwargs) -> None:
        kwargs["sum_square"] += self.eps + np.power(kwargs["grad"], 2)

        # 자주 크게 움직인 좌표는 분모가 커져 학습률이 자연스럽게 줄어든다.
        lr = np.divide(self.lr, np.sqrt(kwargs["sum_square"]))

        kwargs["param"] -= lr * kwargs["grad"]


class RegularizedSGD(Optimizer):
    """L2 regularization 항을 함께 적용하는 SGD."""

    def __init__(self, lr: float = 0.01, alpha: float = 0.1) -> None:
        super().__init__()
        self.lr = lr
        self.alpha = alpha

    def step(self) -> None:
        for (param, param_grad) in zip(self.net.params(), self.net.param_grads()):
            self._update_rule(param=param, grad=param_grad)

    def _update_rule(self, **kwargs) -> None:
        kwargs["param"] -= self.lr * kwargs["grad"] + self.alpha * kwargs["param"]
