from typing import Any, cast

from omegaconf import OmegaConf

from fewsound.cfg import Settings, EncoderType
from fewsound.models.encoders.default_encoder import Encoder
from fewsound.models.encoders.wav2vec import Wav2VecEncoder
from fewsound.models.encoders.hubert import HubertEncoder
from fewsound.models.encoders.snac import Snac
from fewsound.models.encoders.spectrogram_encoder import SpectrogramEncoder


def get_encoder(cfg: Settings) -> "AudioEncoder":
    encoder_type: EncoderType = cfg.model.encoder_type
    if encoder_type == EncoderType.DEFAULT:
        encoder = Encoder(
            C=cfg.model.encoder_channels,
            D=cfg.model.embedding_size,
            **cast(dict[str, Any], OmegaConf.to_container(cfg.model, resolve=True)),
        )
    elif encoder_type == EncoderType.WAV2VEC:
        encoder = Wav2VecEncoder(
            freeze_feature_extractor=cfg.model.freeze_ssl_encoder,
        )
    elif encoder_type == EncoderType.HUBERT:
        encoder = HubertEncoder(
            freeze_feature_extractor=cfg.model.freeze_ssl_encoder,
        )
    elif encoder_type == EncoderType.SPECTRAL_ENCODER:
        encoder = SpectrogramEncoder(
            use_pitch_encoder=cfg.model.use_pitch_encoder,
            use_loudness_encoder=cfg.model.use_loudness_encoder,
            use_phoneme_encoder=cfg.model.use_phoneme_encoder,
        )
    elif encoder_type == EncoderType.SNAC:
        encoder = Snac(
            attn_window_size=16,
            strides=[4, 4, 8, 8],
            d_model=32,
        )
    else:
        raise ValueError("Encoder type not supported")
    return encoder
