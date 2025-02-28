import torch
from torch import nn
from transformers import Wav2Vec2Processor, Wav2Vec2Model

from fewsound.models.encoders.types import AudioEncoder


class Wav2VecEncoder(AudioEncoder):
    def __init__(
        self,
        sampling_rate: int = 16_000, # w2v takes 16kHz as an input
        device: str = "cuda",
        freeze_feature_extractor: bool = True,
    ):
        super().__init__()
        self.model_path = "facebook/wav2vec2-large-960h"
        self.device = device

        self.processor = Wav2Vec2Processor.from_pretrained(self.model_path)
        self.model = Wav2Vec2Model.from_pretrained(self.model_path)
        if freeze_feature_extractor:
            for param in self.model.parameters():
                param.requires_grad = False

        self.gru = nn.GRU(input_size=1024, hidden_size=3296, batch_first=True)
        self.sampling_rate = sampling_rate

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.processor(
            x.tolist(),
            return_tensors="pt",
            sampling_rate=self.sampling_rate,
        ).input_values
        x = x.to(self.device)

        w2v_embedding = self.model(x).last_hidden_state
        _, hn = self.gru(w2v_embedding)
        return hn[-1]

    def output_width(self, input_length: int) -> int:
        return 3296


if __name__ == "__main__":
    device = "cuda"
    model = Wav2VecEncoder(sampling_rate=16_000)
    model = model.to(device)

    x = torch.rand((1, 2 * 16_000))  # 2 second rand audio
    embedding = model(x)

    print(embedding.shape)
