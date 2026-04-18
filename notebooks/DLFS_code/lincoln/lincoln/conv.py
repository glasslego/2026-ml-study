"""NumPy로 구현한 다채널 2D convolution 연산."""

import numpy as np
from numpy import ndarray

from .base import ParamOperation


class Conv2D_Op(ParamOperation):
    """합성곱 필터를 적용하는 파라미터 연산.

    입력 텐서 shape은 `[batch, in_channels, height, width]`,
    필터 텐서 shape은 `[in_channels, out_channels, kernel, kernel]`로 가정한다.
    """

    def __init__(self, W: ndarray):
        super().__init__(W)
        self.param_size = W.shape[2]
        self.param_pad = self.param_size // 2

    def _pad_1d(self, inp: ndarray) -> ndarray:
        """1차원 벡터 양 끝에 0을 붙인다."""

        z = np.array([0])
        z = np.repeat(z, self.param_pad)
        return np.concatenate([z, inp, z])

    def _pad_1d_batch(self, inp: ndarray) -> ndarray:
        """2차원 배열의 각 행에 동일한 1D padding을 적용한다."""

        outs = [self._pad_1d(obs) for obs in inp]
        return np.stack(outs)

    def _pad_2d_obs(self, inp: ndarray):
        """단일 채널 2D 이미지에 상하좌우 0 padding을 적용한다."""

        inp_pad = self._pad_1d_batch(inp)

        other = np.zeros((self.param_pad, inp.shape[0] + self.param_pad * 2))

        return np.concatenate([other, inp_pad, other])

    def _pad_2d_channel(self, inp: ndarray):
        """단일 관측치의 모든 채널에 대해 2D padding을 적용한다."""

        return np.stack([self._pad_2d_obs(channel) for channel in inp])

    def _get_image_patches(self, input_: ndarray):
        """이미지 전체를 슬라이딩 윈도우 patch 묶음으로 바꾼다.

        합성곱은 작은 필터를 이미지 전체에 조금씩 이동시키며 계산한다.
        여기서는 그 과정을 "필터 크기만큼 잘라낸 patch를 모두 모은 뒤,
        행렬곱으로 한 번에 처리"하는 방식으로 구현한다.
        """

        imgs_batch_pad = np.stack([self._pad_2d_channel(obs) for obs in input_])
        patches = []
        img_height = imgs_batch_pad.shape[2]
        for h in range(img_height - self.param_size + 1):
            for w in range(img_height - self.param_size + 1):
                patch = imgs_batch_pad[:, :, h : h + self.param_size, w : w + self.param_size]
                patches.append(patch)
        return np.stack(patches)

    def _output(self, inference: bool = False):
        """순전파 합성곱을 수행한다."""

        batch_size = self.input_.shape[0]
        img_height = self.input_.shape[2]
        img_size = self.input_.shape[2] * self.input_.shape[3]
        patch_size = self.param.shape[0] * self.param.shape[2] * self.param.shape[3]

        patches = self._get_image_patches(self.input_)

        # patch 묶음을 [batch, 위치 개수, patch 내부 원소 수]로 펼치면
        # 뒤에서 필터 행렬과 matmul을 수행할 수 있다.
        patches_reshaped = (
            patches.transpose(1, 0, 2, 3, 4).reshape(batch_size, img_size, -1)
        )

        # 필터 역시 각 출력 채널마다 하나의 긴 벡터로 편다.
        param_reshaped = (
            self.param.transpose(0, 2, 3, 1).reshape(patch_size, -1)
        )

        output_reshaped = (
            np.matmul(patches_reshaped, param_reshaped)
            .reshape(batch_size, img_height, img_height, -1)
            .transpose(0, 3, 1, 2)
        )

        return output_reshaped

    def _input_grad(self, output_grad: np.ndarray) -> np.ndarray:
        """출력 gradient를 입력 gradient로 변환한다."""

        batch_size = self.input_.shape[0]
        img_size = self.input_.shape[2] * self.input_.shape[3]
        img_height = self.input_.shape[2]

        output_patches = (
            self._get_image_patches(output_grad)
            .transpose(1, 0, 2, 3, 4)
            .reshape(batch_size * img_size, -1)
        )

        # 입력 gradient는 "출력 gradient patch"와 "필터 전치"의 곱으로 얻는다.
        param_reshaped = self.param.reshape(self.param.shape[0], -1).transpose(1, 0)

        return (
            np.matmul(output_patches, param_reshaped)
            .reshape(batch_size, img_height, img_height, self.param.shape[0])
            .transpose(0, 3, 1, 2)
        )

    def _param_grad(self, output_grad: ndarray) -> ndarray:
        """필터 파라미터에 대한 gradient를 계산한다."""

        batch_size = self.input_.shape[0]
        img_size = self.input_.shape[2] * self.input_.shape[3]
        in_channels = self.param.shape[0]
        out_channels = self.param.shape[1]

        in_patches_reshape = (
            self._get_image_patches(self.input_)
            .reshape(batch_size * img_size, -1)
            .transpose(1, 0)
        )

        out_grad_reshape = (
            output_grad.transpose(0, 2, 3, 1).reshape(batch_size * img_size, -1)
        )

        return (
            np.matmul(in_patches_reshape, out_grad_reshape)
            .reshape(in_channels, self.param_size, self.param_size, out_channels)
            .transpose(0, 3, 1, 2)
        )
