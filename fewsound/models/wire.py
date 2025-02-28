# Adapted from https://github.com/vishwa91/wire
# license in legal

from typing import Optional, Union

import torch
from torch import Tensor

class GaborWavelet(torch.nn.Module):
    def __init__(self, omega: float, scale: float):
        super().__init__()
        self.omega = omega
        self.scale = scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        omega = self.omega * x
        scale = self.scale * x
        return torch.exp(1j * omega - scale.abs().square())

class WIRE(torch.nn.Module):
    def __init__(
        self,
        input_size: int,
        output_size: int,
        hidden_sizes: list[int],
        omega_0,
        hidden_omega,
        scale,
    ) -> None:
        super().__init__()
        net = [
            torch.nn.Linear(input_size, hidden_sizes[0]),
            GaborWavelet(omega_0, scale)
        ]
        for in_dim, out_dim in zip(hidden_sizes, hidden_sizes[1:]):
            net.append(torch.nn.Linear(in_dim, out_dim, dtype=torch.cfloat))
            net.append(GaborWavelet(hidden_omega, scale))
        net.append(torch.nn.Linear(
            hidden_sizes[-1], output_size, dtype=torch.cfloat)
        )
        self.net = torch.nn.Sequential(*net)
        self.params = {} # for compatibility

    def num_params(self):
        return sum(param.numel() for param in self.parameters())

    def forward(  # type: ignore
            self,
            x: Tensor,
            weights: Optional[dict[str, Tensor]] = None,
            return_activations: bool = False,
    ) -> Union[Tensor, tuple[Tensor, list[tuple[Tensor, Tensor]]]]:
        output = self.net(x)
        output = output.real
        if return_activations:
            return output, [] # dummy, complex numbers are incompatible anyway
        else:
            return output
