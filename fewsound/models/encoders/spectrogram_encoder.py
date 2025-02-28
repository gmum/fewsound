# Spectrogram Encoder originally comes from https://github.com/voiceboxneurips/voicebox
# license in legal
import math
from pathlib import Path
from typing import Optional, Callable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchaudio.transforms import LFCC, MFCC, MelSpectrogram, Spectrogram

from fewsound.models.encoders.aux_encoders import LoudnessEncoder, PitchEncoder
from fewsound.models.encoders.types import AudioEncoder

PPG_ENCODER_PATH = Path(__file__).parent / "assets/causal_ppg_256_hidden_512_hop.pt"

################################################################################
# Phoneme predictor model from AC-VC (Ronssin & Cernak)
################################################################################


class Delta(nn.Module):
    """Causal delta computation"""
    def forward(self, x: torch.Tensor):

        x = F.pad(x, (0, 1))
        x = torch.diff(x, n=1, dim=-1)

        return x


class PPGEncoder(nn.Module):
    """
    Phonetic posteriorgram (PPG) predictor from Almost-Causal Voice Conversion
    """
    def __init__(
        self,
        win_length: int = 256,
        hop_length: int = 128,
        win_func: Callable = torch.hann_window,
        n_mels: int = 32,
        n_mfcc: int = 13,
        lstm_depth: int = 2,
        hidden_size: int = 512,
        sampling_rate: int = 16_000,
    ):
        """
        Parameters
        ----------

        win_length (int):       spectrogram window length in samples

        hop_length (int):       spectrogram hop length in samples

        win_func (Callable):    spectrogram window function

        n_mels (int):           number of mel-frequency bins

        n_mfcc (int):           number of cepstral coefficients

        lstm_depth (int):       number of LSTM layers

        hidden_size (int):      hidden layer dimension for MLP and LSTM
        """

        super().__init__()

        self.win_length = win_length
        self.hop_length = hop_length

        # compute spectral representation
        mel_kwargs = {
            "n_fft": self.win_length,
            "win_length": self.win_length,
            "hop_length": self.hop_length,
            "window_fn": win_func,
            "n_mels": n_mels
        }
        self.mfcc = MFCC(
            sample_rate=sampling_rate,
            n_mfcc=n_mfcc,
            log_mels=True,
            melkwargs=mel_kwargs
        )

        # compute first- and second-order MFCC deltas
        self.delta = Delta()

        # PPG network
        self.mlp = nn.Sequential(
            nn.Linear(n_mfcc * 3, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU()
        )
        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=lstm_depth,
            bias=True,
            batch_first=True,
            bidirectional=False
        )

    def forward(self, x: torch.Tensor):

        # require batch, channel dimensions
        assert x.ndim >= 2
        if x.ndim == 2:
            x = x.unsqueeze(1)

        # convert to mono audio
        x = x.mean(dim=1)

        mfcc = self.mfcc(x)  # (n_batch, n_mfcc, n_frames)
        delta1 = self.delta(mfcc)  # (n_batch, n_mfcc, n_frames)
        delta2 = self.delta(delta1)  # (n_batch, n_mfcc, n_frames)

        x = torch.cat([mfcc, delta1, delta2], dim=1)  # (n_batch, 3 * n_mfcc, n_frames)
        x = x.permute(0, 2, 1)  # (n_batch, n_frames, 3 * n_mfcc)

        x = self.mlp(x)  # (n_batch, n_frames, hidden_size)
        x, _ = self.lstm(x)  # (n_batch, n_frames, hidden_size)

        return x


class FiLM(nn.Module):
    """
    Affine conditioning layer, as proposed in Perez et al.
    (https://arxiv.org/pdf/1709.07871.pdf). Operates on each channel of a
    selected feature representation, with one learned scaling parameter and one
    learned bias parameter per channel.

    Code adapted from https://github.com/csteinmetz1/steerable-nafx
    """

    def __init__(
        self,
        cond_dim: int,
        num_features: int,
        batch_norm: bool = True,
    ):
        """
        Apply linear projection and batch normalization to obtain affine
        conditioning parameters.

        :param cond_dim: dimension of conditioning input
        :param num_features: number of feature maps to which conditioning is
                             applied
        :param batch_norm: if True, perform batch normalization
        """
        super().__init__()
        self.num_features = num_features
        self.batch_norm = batch_norm
        if batch_norm:
            self.bn = BatchNorm(num_features, feature_dim=-1, affine=False)
        self.adaptor = torch.nn.Linear(cond_dim, num_features * 2)

    def forward(self, x: torch.Tensor, cond: torch.Tensor):
        """

        FIGURE OUT SHAPES

        x (Tensor):    shape
        cond (Tensor): shape
        """

        # linear projection of conditioning input
        cond = self.adaptor(cond)

        # learn scale and bias parameters per channel, thus 2X num_features
        g, b = torch.chunk(cond, 2, dim=-1)

        if self.batch_norm:
            x = self.bn(x)  # apply BatchNorm without affine
        x = (x * g) + b  # then apply conditional affine

        return x


class ExponentialUnitNorm(nn.Module):
    """Unit-normalize magnitude spectrogram"""

    def __init__(self, decay: float, hop_size: int, n_freq: int, eps: float = 1e-14, sample_rate: int = 16_000):
        """
        Perform exponential unit normalization on magnitude spectrogram

        Parameters
        ----------
        decay (float):

        hop_size (int):

        n_freq (int):

        eps (float):
        """

        super().__init__()

        # compute exponential decay factor
        self.alpha = self._get_norm_alpha(sample_rate, hop_size, decay)
        self.eps = eps
        self.n_freq = n_freq
        self.init_state: torch.Tensor

        # initialize per-band states for unit norm calculation
        self.reset()

    @staticmethod
    def _get_norm_alpha(sr: int, hop_size: int, decay: float):
        """
        Compute exponential decay factor `alpha` for a given decay window size
        in seconds
        """
        dt = hop_size / sr
        a_ = math.exp(-dt / decay)

        precision = 3
        a = 1.0

        while a >= 1.0:
            a = round(a_, precision)
            precision += 1

        return a

    def reset(self):
        """(Re)-initialize stored state"""
        s = torch.linspace(0.001, 0.0001, self.n_freq).view(
            1, self.n_freq
        )  # broadcast with (n_batch, 1, n_frames, n_freq, 2)
        self.register_buffer("init_state", s)

    def forward(self, x: torch.Tensor):
        """
        Perform exponential unit normalization on magnitude spectrogram

        Parameters
        ----------
        x (Tensor): shape (n_batch, n_freq, n_frames)

        Returns
        -------
        normalized (Tensor): shape (n_batch, n_freq, n_frames)
        """
        x_abs = x.clamp_min(1e-10).sqrt()

        n_batch, n_freq, n_frames = x.shape
        assert n_freq == self.n_freq

        state = self.init_state.clone().expand(n_batch, n_freq)  # (n_batch, n_freq)

        out_states = []
        for f in range(n_frames):
            state = x_abs[:, :, f] * (1 - self.alpha) + state * self.alpha
            out_states.append(state)

        return x / torch.stack(out_states, 2).sqrt()


class BatchNorm(nn.Module):
    """Apply batch normalization along feature dimension only"""

    def __init__(self, num_features, feature_dim: int = -1, **kwargs):
        super().__init__()

        if feature_dim == 1:
            self.permute = (0, 1, 2)
        elif feature_dim in [2, -1]:
            self.permute = (0, 2, 1)
        else:
            raise ValueError(f"Must provide batch-first inputs")

        self.num_features = num_features
        self.feature_dim = feature_dim

        # pass any additional arguments to batch normalization module
        self.bn = nn.BatchNorm1d(num_features=self.num_features, **kwargs)

    def forward(self, x: torch.Tensor):
        # check input dimensions
        assert x.ndim == 3
        assert x.shape[self.feature_dim] == self.num_features

        # reshape to ensure batch normalization is time-distributed
        x = x.permute(*self.permute)

        # apply normalization
        x = self.bn(x)

        # restore original shape
        x = x.permute(*self.permute)

        return x


class MLP(nn.Module):
    """Time-distributed MLP network"""

    def __init__(
        self, in_channels: int, hidden_size: int = 512, depth: int = 2, activation: nn.Module = nn.LeakyReLU()
    ):
        super().__init__()

        channels = [in_channels] + depth * [hidden_size]
        mlp = []
        for i in range(depth):
            mlp.append(nn.Linear(channels[i], channels[i + 1]))
            mlp.append(nn.LayerNorm(channels[i + 1]))

            # omit nonlinearity after final layer
            if i < depth - 1:
                mlp.append(activation)

        self.mlp = nn.Sequential(*mlp)

    def forward(self, x: torch.Tensor):
        return self.mlp(x)


class CausalPadding(nn.Module):
    """Perform 'causal' padding at end of signal along final tensor dimension"""

    def __init__(self, pad: int = 0):
        super().__init__()
        self.pad = pad

    def forward(self, x: torch.Tensor):
        return F.pad(x, (0, self.pad))


class SpectrogramEncoder(AudioEncoder):
    """Spectrogram encoder with optional lookahead"""

    def __init__(
        self,
        win_length: int = 512,
        win_type: str = "hann",
        spec_type: str = "linear",
        lookahead: int = 5,
        hidden_size: int = 512,
        n_features: int = 64,
        mlp_depth: int = 2,
        normalize: Optional[str] = None,
        sample_rate: int = 16_000,
        use_pitch_encoder: bool = False,
        use_loudness_encoder: bool = False,
        use_phoneme_encoder: bool = False,
        ppg_encoder_path: str = PPG_ENCODER_PATH,
    ):
        super().__init__()

        # check validity of attributes
        assert normalize in [None, "none", "instance", "exponential"]
        if win_type not in ["rectangular", "triangular", "hann"]:
            raise ValueError(f"Invalid window type {win_type}")

        # store attributes
        self.win_length = win_length
        self.win_type = win_type
        self.lookahead = lookahead
        self.hidden_size = hidden_size
        self.n_features = n_features
        self.spec_type = spec_type
        self.mlp_depth = mlp_depth
        self.normalize = normalize
        self.sample_rate = sample_rate

        self.use_pitch_encoder = use_pitch_encoder
        self.use_loudness_encoder = use_loudness_encoder
        self.use_phoneme_encoder = use_phoneme_encoder

        # determine hop length from window function
        if self.win_type == "rectangular":  # non-overlapping frames
            self.hop_length = self.win_length
        else:
            self.hop_length = self.win_length // 2

        # determine spectrogram normalization method
        n_freq = n_features if spec_type in ["mel", "mfcc", "lfcc"] else win_length // 2 + 1

        if normalize in [None, "none"]:
            self.norm = nn.Identity()
        elif normalize == "instance":
            self.norm = nn.InstanceNorm1d(num_features=n_freq, track_running_stats=True)
        elif normalize == "exponential":
            self.norm = ExponentialUnitNorm(
                decay=1.0, hop_size=self.hop_length, n_freq=n_freq, sample_rate=sample_rate
            )

        # compute spectral representation
        spec_kwargs = {
            "n_fft": self.win_length,
            "win_length": self.win_length,
            "hop_length": self.hop_length,
            "window_fn": self._get_win_func(self.win_type),
        }
        mel_kwargs = {**spec_kwargs, "n_mels": self.n_features}

        if spec_type == "linear":
            self.spec = Spectrogram(**spec_kwargs)
        elif spec_type == "mel":
            self.spec = MelSpectrogram(sample_rate=sample_rate, **mel_kwargs)
        elif spec_type == "mfcc":
            self.spec = MFCC(sample_rate=sample_rate, n_mfcc=self.n_features, log_mels=True, melkwargs=mel_kwargs)
        elif spec_type == "lfcc":
            mel_kwargs.pop("n_mels")
            self.spec = LFCC(sample_rate=sample_rate, n_lfcc=self.n_features, speckwargs=mel_kwargs)

        # GLU - learn which channels of input to pass through most strongly
        self.glu = nn.Sequential(
            nn.Conv1d(in_channels=n_freq, out_channels=self.hidden_size * 2, kernel_size=1, stride=1), nn.GLU(dim=1)
        )

        # Conv1D layers
        conv = []
        for i in range(lookahead):
            conv.extend(
                [
                    CausalPadding(1),
                    nn.Conv1d(in_channels=self.hidden_size,out_channels=self.hidden_size, kernel_size=2, stride=1),
                    BatchNorm(num_features=self.hidden_size, feature_dim=1) if i < lookahead - 1 else nn.Identity(),
                    nn.ReLU(),
                ]
            )
        self.conv = nn.Sequential(*conv)

        aux_encoder_channels = 259 # 256 x PPG, 2x pitch, loudness
        if use_phoneme_encoder:
            self.ppg_encoder = PPGEncoder(
                win_length=1024,
                hop_length=1024//2,
                hidden_size=256,
                n_mfcc=19
            )
            self.ppg_encoder.load_state_dict(
                torch.load(ppg_encoder_path, map_location=torch.device('cpu'))
            )
            for param in self.ppg_encoder.parameters():
                param.requires_grad = False
        else:
            self.ppg_encoder = nn.Identity()

        if use_pitch_encoder:
            self.pitch_encoder = PitchEncoder(
                algorithm="dio",
                return_periodicity=True,
                hop_length=self.hop_length,
                sample_rate=self.sample_rate
            )
            for param in self.pitch_encoder.parameters():
                param.requires_grad = False
        else:
            self.pitch_encoder = nn.Identity()

        if use_loudness_encoder:
            self.loudness_encoder = LoudnessEncoder(
                hop_length=self.hop_length,
                n_fft=self.n_features,
                sample_rate=self.sample_rate,
            )
            for param in self.loudness_encoder.parameters():
                param.requires_grad = False
        else:
            self.loudness_encoder = nn.Identity()


        n_encoder_feats = hidden_size + aux_encoder_channels
        self.encoder_proj = nn.Sequential(
            BatchNorm(num_features=n_encoder_feats),
            nn.Linear(n_encoder_feats, self.hidden_size),
            nn.ReLU()
        )

        # pre-bottleneck MLP
        self.mlp = MLP(in_channels=self.hidden_size, hidden_size=self.hidden_size, depth=mlp_depth)
        self.gru = nn.GRU(input_size=self.hidden_size, hidden_size=3296, batch_first=True)

    @staticmethod
    def _get_win_func(win_type: str):
        if win_type == "rectangular":
            return lambda m: torch.ones(m)
        elif win_type == "hann":
            return lambda m: torch.hann_window(m)
        elif win_type == "triangular":
            return lambda m: torch.as_tensor(np.bartlett(m)).float()

    def forward(self, x: torch.Tensor):
        # ensure the input contains at least a single frame of audio
        assert x.shape[-1] >= self.win_length

        # require batch, channel dimensions
        assert x.ndim >= 2
        if x.ndim == 2:
            x = x.unsqueeze(1)

        # convert to mono audio
        x = x.mean(dim=1)

        # compute spectrogram
        spec = self.spec(x) + 1e-6  # (n_batch, n_freq, n_frames)

        if self.spec_type in ["linear", "mel"]:
            spec = 10 * torch.log10(spec + 1e-8)  # (n_batch, n_freq, n_frames)

        # normalize spectrogram
        spec = self.norm(spec)  # (n_batch, n_freq, n_frames)

        # actual encoder network
        encoded = self.glu(spec)  # (n_batch, hidden_size, n_frames)
        encoded = self.conv(encoded)  # (n_batch, hidden_size, n_frames)
        encoded = self.mlp(encoded.permute(0, 2, 1))  # (n_batch, n_frames, hidden_size)

        features = [encoded]
        # join with phoneme, loudness and pitch estimates (they are not optimized)

        if self.use_phoneme_encoder:
            ppgs = self.ppg_encoder(x)
            # PPGs return 2x smaller size - repeat them
            ppgs = torch.repeat_interleave(ppgs, repeats=2, dim=1)
            ppgs = ppgs[:, :encoded.shape[1], :] # in case of the odd shapes
            features.append(ppgs)

        if self.use_loudness_encoder:
            loudness = self.loudness_encoder(x)
            features.append(loudness)

        if self.use_pitch_encoder:
            pitch, periodicity = self.pitch_encoder(x)
            features += [pitch, periodicity]

        features = torch.cat(features, dim=-1)  # (n_batch, n_frames, hidden_size + n)
        encoded = self.encoder_proj(features)

        _, hn = self.gru(encoded)
        encoded = hn[-1]

        return encoded

    def output_width(self, input_length: int) -> int:
        return 3296


if __name__ == "__main__":
    # SpectrogramEncoder

    model = SpectrogramEncoder(use_pitch_encoder=False, use_loudness_encoder=True)

    x = torch.rand((1, 2 * 16_000))  # 2 second rand audio
    embedding = model(x)

    print(embedding.shape)
