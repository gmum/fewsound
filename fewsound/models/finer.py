# FINER arxiv: https://arxiv.org/abs/2312.02434

# code adapted from https://github.com/gmum/FreSh/blob/main/src/models.py

from typing import Optional, Union, cast

import torch
import torch.nn
from torch import Tensor
from torch.nn.parameter import Parameter

from fewsound.cfg import TargetNetworkMode
from fewsound.models.meta.inr import INR

from fewsound.models.siren import Sine


class FINER(INR):
    def __init__(
        self,
        input_size: int,
        output_size: int,
        hidden_sizes: list[int],
        omega_0: float,
        omega_i: float,
        bias_scale: float,
        scale_requires_grad: bool = False
    ):
        super().__init__(
            input_size=input_size,
            output_size=output_size,
            hidden_sizes=hidden_sizes,
            bias=True,
            activation_fn=Sine(),
            mode=TargetNetworkMode.INR, # only use as INR
        )

        for i in range(self.n_layers):
            omega_val = torch.ones((1,), dtype=torch.float32)
            if i == 0:
                omega_val *= omega_0
            else:
                omega_val *= omega_i
            omega = Parameter(omega_val, requires_grad=False)
            self.params[f"o{i}"] = omega
            self.register_parameter(f"o{i}", omega)

        for i in range(self.n_layers):
            w_i = self.params[f"w{i}"]
            _, n_features_in = w_i.shape

            if i == 0:
                std = 1 / n_features_in
            else:
                std = ((6.0 / n_features_in) ** 0.5) / omega_i

            with torch.no_grad():
                w_i.uniform_(-std, std)

        with torch.no_grad():
            first_bias = self.params["b0"]
            first_bias.uniform_(-bias_scale, bias_scale)

        self.scale_req_grad = scale_requires_grad

    def _get_scale(self, x: Tensor) -> Tensor:
        if self.scale_req_grad:
            scale = torch.abs(x) + 1
        else:
            with torch.no_grad():
                scale = torch.abs(x) + 1
        return scale

    def forward(  # type: ignore
        self,
        x: Tensor,
        weights: Optional[dict[str, Tensor]] = None,
        return_activations: bool = False,
    ) -> Union[Tensor, tuple[Tensor, list[tuple[Tensor, Tensor]]]]:

        activations: list[tuple[Tensor, Tensor]] = []

        for i in range(self.n_layers):
            omega = cast(Tensor, self.params[f"o{i}"])

            scale = self._get_scale(x)
            x = omega * scale * x
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
