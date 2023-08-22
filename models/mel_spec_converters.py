from dataclasses import asdict

import torch
from torchaudio.transforms import MelSpectrogram

from utils.containers import MelSpecParameters
from .base import MelSpecConverter


class SimpleMelSpecConverter(MelSpecConverter):
    def __init__(self, mel_spec_params: MelSpecParameters) -> None:
        self.mel_spec_params = mel_spec_params
        self.mel_spec = MelSpectrogram(**asdict(mel_spec_params))

    def convert(self, slice: torch.Tensor) -> torch.Tensor:
        output = self.mel_spec(slice)
        return output


class ScaledImageMelSpecConverter(MelSpecConverter):
    def __init__(self, mel_spec_params: MelSpecParameters) -> None:
        self.mel_spec_params = mel_spec_params
        self.mel_spec = MelSpectrogram(**asdict(mel_spec_params))

    def convert(self, slice: torch.Tensor) -> torch.Tensor:
        output = self.mel_spec(slice)
        scaled_output = torch.tanh(output)

        # Create the repeat tensor
        scaled_output = torch.cat(
            [
                scaled_output.clone(),
                scaled_output.clone(),
                scaled_output.clone(),
            ],
            dim=-3,
        )

        return scaled_output
