# fourier-features arxiv: https://arxiv.org/abs/2006.10739

from typing import Optional, Union

import torch
import torch.nn
from torch import Tensor, nn
import rff.layers

from fewsound.cfg import TargetNetworkMode
from fewsound.models.meta.inr import INR


class RFF(INR):
    def __init__(
        self,
        input_size: int,
        output_size: int,
        hidden_sizes: list[int],
        sigma: float
    ):
        super().__init__(
            input_size=hidden_sizes[0],
            output_size=output_size,
            hidden_sizes=hidden_sizes,
            bias=True,
            activation_fn=nn.ReLU(),
            mode=TargetNetworkMode.INR,
        )
        self.input_size = input_size  # override super().input_size
        self.encoding = rff.layers.GaussianEncoding(
            sigma=sigma, input_size=input_size, encoded_size=hidden_sizes[0] // 2
        )

    def forward(  # type: ignore
        self,
        x: Tensor,
        weights: Optional[dict[str, Tensor]] = None,
        return_activations: bool = False,
    ) -> Union[Tensor, tuple[Tensor, list[tuple[Tensor, Tensor]]]]:

        x = self.encoding(x)

        activations: list[tuple[Tensor, Tensor]] = []

        for i in range(self.n_layers):
            x = self._forward(x, layer_idx=i, weights=weights)

            h = x

            if i != self.n_layers - 1:
                x = self._activation_fn(x)

            if return_activations:
                activations.append((x, h))

        if return_activations:
            return x, activations
        else:
            return x
