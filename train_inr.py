import copy
from pathlib import Path
from typing import Optional, Type, cast

import hydra
import pytorch_lightning as pl
import pytorch_yard
import torch
import torch.utils.data
from omegaconf import OmegaConf
from omegaconf.dictconfig import DictConfig
from pytorch_yard.configs import get_tags
from pytorch_yard.experiments.lightning import LightningExperiment
from torch import Tensor
from torch.utils.data import RandomSampler, TensorDataset
from torchaudio.transforms import MelSpectrogram, AmplitudeToDB

from fewsound.cfg import Settings
from fewsound.datasets.utils import init_datamodule
from fewsound.systems.main import HyperNetworkAE
from fewsound.utils.metrics import reduce_metric
from inr.systems.main import INRSystem

import librosa
import numpy as np
from torch import nn

class AudioFile(torch.utils.data.Dataset):
    def __init__(self, filename):
        data, rate = librosa.load(filename, sr=None)
        self.audio = torch.Tensor(data / np.max(np.abs(data)))

        indices = torch.arange(0, len(data), dtype=torch.float).unsqueeze(-1)
        min_val, max_val = -1, 1
        self.indices = min_val + (max_val - min_val) * indices / (len(data) - 1)

        mel_spec = MelSpectrogram(sample_rate=rate, n_fft=1024, hop_length=256, n_mels=128)
        atdb = AmplitudeToDB()
        self.spectrogram = atdb(mel_spec(self.audio)).unsqueeze(-3)

    def get_num_samples(self):
        return len(self.audio)

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        return self.audio, self.indices, self.spectrogram


class SingleINRExperiment(LightningExperiment):
    def __init__(self, config_path: str, settings_cls: Type[Settings], settings_group: Optional[str] = None) -> None:
        super().__init__(config_path, settings_cls, settings_group=settings_group)

        self.cfg: Settings
        """ Experiment config. """

    def entry(self, root_cfg: pytorch_yard.RootConfig):
        super().entry(root_cfg)

    # Do not use pytorch-yard template specializations as we use a monolithic `main` here.
    def setup_system(self):
        pass

    def setup_datamodule(self):
        pass

    # ------------------------------------------------------------------------
    # Experiment specific code
    # ------------------------------------------------------------------------
    def main(self):
        # --------------------------------------------------------------------
        # W&B init
        # --------------------------------------------------------------------
        tags: list[str] = get_tags(cast(DictConfig, self.root_cfg))
        self.run.tags = tags
        self.run.notes = str(self.root_cfg.notes)
        self.wandb_logger.log_hyperparams(OmegaConf.to_container(self.root_cfg.cfg, resolve=True))  # type: ignore

        # --------------------------------------------------------------------
        # Data module setup
        # --------------------------------------------------------------------
        Path(self.root_cfg.data_dir).mkdir(parents=True, exist_ok=True)

        self.root_cfg.cfg = cast(Settings, self.root_cfg.cfg)
        self.root_cfg.cfg.batch_size = 1
        self.root_cfg.cfg.save_checkpoints = False

        self.datamodule, _ = init_datamodule(self.root_cfg)
        self.datamodule.prepare_data()
        self.datamodule.setup()

        # --------------------------------------------------------------------
        # Trainer setup
        # --------------------------------------------------------------------
        self.setup_callbacks()

        combined_metrics: list[dict[str, Tensor]] = []


        callbacks = copy.deepcopy(self.callbacks)

        self.trainer: pl.Trainer = hydra.utils.instantiate(  # type: ignore
            self.cfg.pl,
            logger=self.wandb_logger,
            callbacks=callbacks,
            enable_checkpointing=False,
            num_sanity_val_steps=0,
        )

        audio, indices, spectrogram = AudioFile(self.cfg.inr_audio_path)[0]
        indices = indices.unsqueeze(0)
        audio = audio.unsqueeze(0)
        spectrograms = spectrogram.unsqueeze(0)
        dataset = TensorDataset(indices, audio, spectrograms)

        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=1,
            sampler=RandomSampler(dataset, replacement=True, num_samples=100),
            num_workers=0,
        )

        self.system = INRSystem(
            cfg=self.cfg,
            spec_transform=copy.deepcopy(self.datamodule.train.spec_transform),
            idx=0,
            extended_logging=True,
        )
        self.trainer.fit(  # type: ignore
            self.system,
            train_dataloaders=dataloader,
        )
        combined_metrics.append(self.system.metrics)


if __name__ == "__main__":
    SingleINRExperiment("fewsound", Settings)
