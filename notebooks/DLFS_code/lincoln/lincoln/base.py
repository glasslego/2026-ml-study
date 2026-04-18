"""모든 연산 레이어가 공유하는 기본 forward/backward 뼈대."""

from numpy import ndarray

from lincoln.utils.np_utils import assert_same_shape


class Operation(object):
    """입력 텐서를 받아 출력 텐서로 바꾸는 가장 기본적인 연산 단위.

    이 클래스는 "순전파 때 필요한 중간값을 저장해 두고, 역전파 때 다시 꺼내 쓴다"는
    패턴을 공통으로 제공한다. 실제 계산식은 하위 클래스가 `_output`,
    `_input_grad`에서 구현한다.
    """

    def __init__(self):
        pass

    def forward(
        self,
        input_: ndarray,
        inference: bool = False,
    ) -> ndarray:
        """순전파를 수행하고 역전파에 필요한 입력/출력을 저장한다.

        Args:
            input_: 현재 연산으로 들어오는 배치 텐서.
            inference: 추론 모드 여부. dropout처럼 학습/추론 동작이 다른 연산에서 쓴다.

        Returns:
            현재 연산이 만든 출력 텐서.
        """

        # 역전파에서는 "이 연산에 어떤 입력이 들어왔는가?"를 다시 알아야 하므로 저장한다.
        self.input_ = input_

        # 실제 계산은 하위 클래스가 담당한다.
        self.output = self._output(inference)

        return self.output

    def backward(self, output_grad: ndarray) -> ndarray:
        """현재 연산의 입력 쪽 gradient를 계산한다.

        Args:
            output_grad: 다음 연산 또는 loss에서 넘어온 출력 기준 gradient.

        Returns:
            현재 연산의 입력 기준 gradient.
        """

        # 체인 룰을 적용하려면 output_grad의 shape이 순전파 출력과 같아야 한다.
        assert_same_shape(self.output, output_grad)

        # 하위 클래스가 정의한 미분 공식을 이용해 입력 쪽 gradient를 구한다.
        self.input_grad = self._input_grad(output_grad)

        # 역전파 결과는 원래 입력 텐서와 같은 shape여야 한다.
        assert_same_shape(self.input_, self.input_grad)

        return self.input_grad

    def _output(self, inference: bool) -> ndarray:
        """실제 순전파 계산을 담당하는 훅 메서드."""

        raise NotImplementedError()

    def _input_grad(self, output_grad: ndarray) -> ndarray:
        """출력 gradient를 입력 gradient로 바꾸는 훅 메서드."""

        raise NotImplementedError()


class ParamOperation(Operation):
    """학습 가능한 파라미터를 가진 연산.

    예를 들어 `WeightMultiply`, `BiasAdd`, `Conv2D_Op`는 입력뿐 아니라
    가중치/편향에 대한 gradient도 계산해야 한다. 그래서 일반 Operation에
    `param`과 `param_grad` 개념을 추가한 클래스가 필요하다.
    """

    def __init__(self, param: ndarray) -> ndarray:
        super().__init__()
        self.param = param

    def backward(self, output_grad: ndarray) -> ndarray:
        """입력 gradient와 파라미터 gradient를 동시에 계산한다."""

        assert_same_shape(self.output, output_grad)

        # 입력 쪽 gradient는 이전 layer로 흘려보내는 데 사용된다.
        self.input_grad = self._input_grad(output_grad)

        # 파라미터 gradient는 optimizer가 가중치를 갱신하는 데 사용된다.
        self.param_grad = self._param_grad(output_grad)

        assert_same_shape(self.input_, self.input_grad)

        return self.input_grad

    def _param_grad(self, output_grad: ndarray) -> ndarray:
        """출력 gradient로부터 파라미터 gradient를 계산한다."""

        raise NotImplementedError()
