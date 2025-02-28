from abc import ABC, abstractmethod

import torch.nn as nn


class AudioEncoder(nn.Module, ABC):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def forward(self, x):
        pass

    @abstractmethod
    def output_width(self):
        pass
