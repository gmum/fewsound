# Adapted from https://github.com/Blealtan/efficient-kan
# License is in legal/

import math
from typing import Optional, Union, cast, List

import torch
import torch.nn
from torch import Tensor, nn
from torch.nn.parameter import Parameter

from fewsound.cfg import TargetNetworkMode


class KAN(nn.Module):
    def __init__(
        self,
        input_size: int,
        output_size: int,
        hidden_sizes: list[int],
        encoding_length: int,
        learnable_encoding: bool,
        mode: TargetNetworkMode,
        grid_size: int = 5,
        spline_order: int = 3,
        scale_noise: float = 0.1,
        scale_base: float = 1.0,
        scale_spline: float = 1.0,
        enable_standalone_scale_spline: bool = True,
        base_activation=torch.nn.SiLU,
        grid_eps: float = 0.02,
        grid_range: List[int] = [-1, 1],
    ):
        super().__init__()
        self.input_size = input_size
        self.encoding_length = encoding_length
        self.mode = mode
        self.n_layers = 1 + len(hidden_sizes)
        self.params: dict[str, Parameter] = {}

        freq = torch.ones((encoding_length,), dtype=torch.float32)
        for i in range(len(freq)):
            freq[i] = 2**i
        freq = freq * torch.pi
        freq = Parameter(freq, requires_grad=learnable_encoding)

        self.params["freq"] = freq

        self.grid_size = grid_size
        self.spline_order = spline_order

        h = (grid_range[1] - grid_range[0]) / grid_size
        grid = (
            (torch.arange(-spline_order, grid_size + spline_order + 1) * h + grid_range[0])
            .expand(input_size, -1)
            .contiguous()
        )
        self.register_buffer("grid", grid)

        self.scale_noise = scale_noise
        self.scale_base = scale_base
        self.scale_spline = scale_spline
        self.enable_standalone_scale_spline = enable_standalone_scale_spline
        self.base_activation = base_activation()
        self.grid_eps = grid_eps

        input_size = 2*encoding_length*input_size
        for i, (n_in, n_out) in enumerate(zip([input_size] + hidden_sizes, hidden_sizes + [output_size])):
            base_weight = torch.nn.Parameter(torch.empty(n_out, n_in), requires_grad=True)
            nn.init.kaiming_uniform_(base_weight, a=math.sqrt(5) * self.scale_base)
            self.params[f"bw{i}"] = base_weight

            spline_weight = torch.nn.Parameter(
                torch.Tensor(n_out, n_in, grid_size + spline_order),
                requires_grad=True
            )
            if enable_standalone_scale_spline:
                spline_scaler = torch.nn.Parameter(torch.Tensor(n_out, n_in))

            with torch.no_grad():
                noise = (
                        (torch.rand(self.grid_size + 1, n_in, n_out) - 1 / 2)
                        * self.scale_noise
                        / self.grid_size
                )
                spline_weight.data.copy_(
                    (self.scale_spline if not self.enable_standalone_scale_spline else 1.0)
                    * self.curve2coeff(
                        self.grid.T[self.spline_order: -self.spline_order],
                        noise,
                        n_in,
                        n_out
                    )
                )
                if enable_standalone_scale_spline:
                    torch.nn.init.kaiming_uniform_(spline_scaler, a=math.sqrt(5) * self.scale_spline)

            self.params[f"sw{i}"] = spline_weight
            if enable_standalone_scale_spline:
                self.params[f"ss{i}"] = spline_scaler

        for name, param in self.params.items():
            self.register_parameter(name, param)

    def num_params(self, shared_params: Optional[list[str]] = None) -> int:
        num_params = 0
        if shared_params is None:
            shared_params = []

        learnable_params = {
            param_name: param
            for param_name, param in self.params.items()
            if param.requires_grad and param_name not in shared_params
        }

        for param in learnable_params.values():
            num_params += param.numel()

        return num_params

    def curve2coeff(self, x: torch.Tensor, y: torch.Tensor, n_in: int, n_out: int):
        """
        Compute the coefficients of the curve that interpolates the given points.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_features).
            y (torch.Tensor): Output tensor of shape (batch_size, in_features, out_features).

        Returns:
            torch.Tensor: Coefficients tensor of shape (out_features, in_features, grid_size + spline_order).
        """

        A = self.b_splines(x).transpose(0, 1)  # (in_features, batch_size, grid_size + spline_order)
        B = y.transpose(0, 1)  # (in_features, batch_size, out_features)
        solution = torch.linalg.lstsq(A, B).solution  # (in_features, grid_size + spline_order, out_features)
        result = solution.permute(2, 0, 1)  # (out_features, in_features, grid_size + spline_order)

        assert result.size() == (
            n_out,
            n_in,
            self.grid_size + self.spline_order,
        )
        return result.contiguous()

    def freeze_params(self, shared_params: list[str]) -> None:  # TODO: Verify
        assert self.mode is TargetNetworkMode.TARGET_NETWORK
        for param_name, param in self.params.items():
            if param_name not in shared_params:
                param.requires_grad = False

    def forward(  # type: ignore
        self,
        x: Tensor,
        weights: Optional[dict[str, Tensor]] = None,
        return_activations: bool = False,
    ) -> Union[Tensor, tuple[Tensor, list[tuple[Tensor, Tensor]]]]:
        if self.mode == TargetNetworkMode.INR:
            x = x.unsqueeze(dim=0)

        if weights is None:
            # Single mode, x --> (num_samples, input_size)
            freq = cast(Tensor, self.params["freq"])  # (encoding_length,)
            weights = self.params
        else:
            # Batch mode, x --> (batch_size, num_samples, input_size)
            freq = weights.get("freq", cast(Tensor, self.params["freq"]).tile(x.shape[0], 1))
            freq = freq.unsqueeze(1).unsqueeze(1)  # (batch_size, _, _, encoding_length)

        x = x.unsqueeze(-1) * freq
        x = torch.cat((torch.sin(x), torch.cos(x)), dim=-1).flatten(-2, -1)
        # x --> (_batch_size_, num_samples, input_size * encoding_length * 2)

        activations: list[tuple[Tensor, Tensor]] = []

        for i in range(self.n_layers):
            spline_weight = weights.get(
                f"sw{i}", torch.stack(x.shape[0] * [cast(Tensor, self.params[f"sw{i}"])], dim=0)
            )
            if self.mode == TargetNetworkMode.INR:
                spline_weight = spline_weight.unsqueeze(0)

            b_splines = self.b_splines(x).view(x.size(0), x.size(1), -1)

            if self.enable_standalone_scale_spline:
                spline_scaler = weights.get(
                    f"ss{i}", torch.stack(x.shape[0] * [cast(Tensor, self.params[f"ss{i}"])], dim=0)
                )
                if self.mode == TargetNetworkMode.INR:
                    spline_scaler = spline_scaler.unsqueeze(0)
                spline_weight = spline_weight * spline_scaler.unsqueeze(-1)

            spline_weight = spline_weight.view(
                spline_weight.size(0),
                spline_weight.size(1),
                -1,
            )
            spline_output = torch.bmm(b_splines, spline_weight.permute(0, 2, 1))

            base_weight = weights.get(
                f"bw{i}",  torch.stack(x.shape[0] * [cast(Tensor, self.params[f"bw{i}"])], dim=0)
            )
            if self.mode == TargetNetworkMode.INR:
                base_weight = base_weight.unsqueeze(0)
            base_output = torch.bmm(self.base_activation(x), base_weight.permute(0, 2, 1))

            x = base_output + spline_output

            h = x

            if return_activations:
                activations.append((x, h))

        if self.mode == TargetNetworkMode.INR:
            x = x.squeeze(dim=0)

        if return_activations:
            return x, activations
        else:
            return x

    def __call__(  # type: ignore
        self,
        x: Tensor,
        weights: Optional[dict[str, Tensor]] = None,
        *,
        return_activations: bool = False,
    ) -> Union[Tensor, tuple[Tensor, list[tuple[Tensor, Tensor]]]]:
        return super().__call__(x, weights, return_activations=return_activations)

    def b_splines(self, x: torch.Tensor):
        """
        Compute the B-spline bases for the given input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_features).

        Returns:
            torch.Tensor: B-spline bases tensor of shape (batch_size, in_features, grid_size + spline_order).
        """

        grid: torch.Tensor = self.grid  # (in_features, grid_size + 2 * spline_order + 1)
        x = x.unsqueeze(-1)
        bases = ((x >= grid[:, :-1]) & (x < grid[:, 1:])).to(x.dtype)
        for k in range(1, self.spline_order + 1):
            bases = (
                (
                    (x - grid[:, : -(k + 1)]) / (grid[:, k:-1] - grid[:, : -(k + 1)])
                    * (bases[:, :, :, :-1] if bases.dim() == 4 else bases[:, :, :-1])
                )
                + (
                    (grid[:, k + 1:] - x) / (grid[:, k + 1:] - grid[:, 1:(-k)])
                    * (bases[:, :, :, 1:] if bases.dim() == 4 else bases[:, :, 1:])
                )
            )
        return bases.contiguous()
