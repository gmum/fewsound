from pathlib import Path

import torch
import torch.nn as nn
from transformers import AutoProcessor, HubertModel

from fewsound.models.encoders.types import AudioEncoder



class HubertEncoder(AudioEncoder):
    def __init__(
        self,
        sampling_rate: int = 16_000,
        device: str = "cuda",
        freeze_feature_extractor: bool = True,
    ):
        super().__init__()
        self.model_path = "facebook/hubert-large-ls960-ft"
        self.device = device

        self.processor = AutoProcessor.from_pretrained(self.model_path)
        self.feature_extractor = HubertModel.from_pretrained(self.model_path)
        if freeze_feature_extractor:
            for param in self.feature_extractor.parameters():
                param.requires_grad = False

        self.gru = nn.GRU(input_size=1024, hidden_size=3296, batch_first=True)
        self.sampling_rate = sampling_rate
        self.device = device

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.processor(
            x,
            return_tensors="pt",
            sampling_rate=self.sampling_rate,
        ).input_values[0, :, :]
        x = x.to(self.device)
        hubert_embedding = self.feature_extractor(x).last_hidden_state
        _, hn = self.gru(hubert_embedding)

        return hn[-1]

    def output_width(self, input_length: int) -> int:
        return 3296


if __name__ == "__main__":
    device = "cuda"
    model = HubertEncoder(sampling_rate=16_000)
    model = model.to(device)

    x = torch.rand((1, 2 * 16_000))  # 2 second rand audio
    embedding = model(x)

    print(embedding.shape)
