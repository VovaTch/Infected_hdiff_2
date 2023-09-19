from dataclasses import asdict
from typing import Protocol

import torch
from torchaudio.transforms import MelSpectrogram

from utils.containers import MelSpecParameters


class MelSpecConverter(Protocol):
    mel_spec: MelSpectrogram

    def __init__(self, mel_spec_params: MelSpecParameters) -> None:
        """Mel spectrogram converter constructor

        Args:
            mel_spec_params (MelSpecParameters): Mel spectrogram parameter object
        """
        ...

    def convert(self, slice: torch.Tensor) -> torch.Tensor:
        """
        Convert a torch tensor representing a music slice to a mel spectrogram

        Args:
            slice (torch.Tensor): Music slice

        Returns:
            torch.Tensor: Mel spectrogram
        """
        ...


class SimpleMelSpecConverter:
    mel_spec: MelSpectrogram

    def __init__(self, mel_spec_params: MelSpecParameters) -> None:
        self.mel_spec_params = mel_spec_params
        self.mel_spec = MelSpectrogram(**asdict(mel_spec_params))

    def convert(self, slice: torch.Tensor) -> torch.Tensor:
        output = self.mel_spec(slice)
        return output


class ScaledImageMelSpecConverter:
    mel_spec: MelSpectrogram

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


MEL_SPEC_CONVERTERS: dict[str, type[MelSpecConverter]] = {
    "simple": SimpleMelSpecConverter,
    "scaled_image": ScaledImageMelSpecConverter,
}
