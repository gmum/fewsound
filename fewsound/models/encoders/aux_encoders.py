# Code adapted from https://github.com/voiceboxneurips/voicebox
# license in legal
import librosa as li
import numpy as np
import torch
import torch.nn as nn

import pyworld

################################################################################
# Compute frame-wise pitch and periodicity estimates
################################################################################


class PitchEncoder(nn.Module):
    def __init__(
        self, algorithm: str = "dio", return_periodicity: bool = True, hop_length: int = 128, sample_rate: int = 22_050
    ):
        super().__init__()

        self.algorithm = algorithm
        self.return_periodicity = return_periodicity
        self.hop_length = hop_length
        self.sample_rate = sample_rate

    @torch.no_grad()
    def forward(self, x: torch.Tensor):
        # ensure the input contains at least a single frame of audio
        assert x.shape[-1] >= self.hop_length

        # avoid modifying input audio
        n_batch, *channel_dims, signal_len = x.shape

        # add channel dimension if necessary
        if len(channel_dims) == 0:
            x = x.unsqueeze(1)

        x = x.mean(dim=1)

        if self.algorithm != "dio":
            raise ValueError(f"Invalid algorithm {self.algorithm}")

        pitch_out, periodicity_out, device = [], [], x.device
        hop_ms = 1000 * self.hop_length / self.sample_rate
        x_np = x.clone().double().cpu().numpy()

        for i in range(n_batch):
            pitch, timeaxis = pyworld.dio(
                x_np[i], fs=self.sample_rate, f0_floor=50, f0_ceil=550, frame_period=hop_ms, speed=4
            )  # downsampling factor, for speedup
            pitch = pyworld.stonemask(
                x_np[i],
                pitch,
                timeaxis,
                self.sample_rate,
            )

            pitch_out.append(pitch)

            if self.return_periodicity:
                unvoiced = pyworld.d4c(
                    x_np[i],
                    pitch,
                    timeaxis,
                    self.sample_rate,
                ).mean(axis=1)

                periodicity_out.append(unvoiced)

        pitch_out = torch.as_tensor(np.stack(pitch_out, axis=0), dtype=torch.float32, device=device).unsqueeze(-1)

        if not self.return_periodicity:
            # (n_batch, n_frames, 1)
            return pitch_out
        else:
            periodicity_out = torch.as_tensor(
                np.stack(periodicity_out, axis=0), dtype=torch.float32, device=device
            ).unsqueeze(
                -1
            )  # remove unsqueeze if not averaging!

            # (n_batch, n_frames), (n_batch, n_frames, 1)
            return pitch_out, periodicity_out


################################################################################
# Extract frame-wise A-weighted loudness
################################################################################


class LoudnessEncoder(nn.Module):
    """Extract frame-wise A-weighted loudness"""

    def __init__(self, hop_length: int = 128, n_fft: int = 256, sample_rate: int = 16_000):
        super().__init__()

        self.hop_length = hop_length
        self.n_fft = n_fft
        self.sample_rate = sample_rate

    def forward(self, x: torch.Tensor):
        n_batch, *channel_dims, signal_len = x.shape

        # require batch, channel dimensions
        assert x.ndim >= 2
        if x.ndim == 2:
            x = x.unsqueeze(1)

        # convert to mono audio
        x = x.mean(dim=1)

        spec = li.stft(
            x.detach().cpu().numpy(),
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.n_fft,
            center=True,
        )
        spec = np.log(abs(spec) + 1e-7)

        # compute A-weighting curve for frequencies of analysis
        f = li.fft_frequencies(sr=self.sample_rate, n_fft=self.n_fft)
        a_weight = li.A_weighting(f)

        # apply multiplicative weighting via addition in log domain
        spec = spec + a_weight.reshape(1, -1, 1)

        # take mean over each frame
        loudness = torch.from_numpy(np.mean(spec, 1)).unsqueeze(-1).float().to(x.device)

        return loudness
