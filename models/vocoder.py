from typing import Any, Dict

from diffwave.model import DiffWave
from diffwave.params import params, AttrDict
import numpy as np
import torch
import torch.nn as nn

from utils.containers import (
    LearningParameters,
    MelSpecParameters,
    MusicDatasetParameters,
)


class DiffwaveWrapper(nn.Module):
    def __init__(
        self,
        learning_params: LearningParameters,
        mel_spec_params: MelSpecParameters,
        dataset_params: MusicDatasetParameters,
    ) -> None:
        self.params = AttrDict(  # Training params
            batch_size=learning_params.batch_size,
            learning_rate=learning_params.learning_rate,
            max_grad_norm=None,
            # Data params
            sample_rate=dataset_params.sample_rate,
            n_mels=mel_spec_params.n_mels,
            n_fft=mel_spec_params.n_fft,
            hop_samples=mel_spec_params.hop_length,
            crop_mel_frames=62,  # Probably an error in paper.
            # Model params
            residual_layers=30,
            residual_channels=64,
            dilation_cycle_length=10,
            unconditional=False,
            noise_schedule=np.linspace(1e-4, 0.05, 50).tolist(),
            inference_noise_schedule=[0.0001, 0.001, 0.01, 0.05, 0.2, 0.5],
            # unconditional sample len
            audio_len=dataset_params.slice_length,  # unconditional_synthesis_samples)
        )
        self.model = DiffWave(AttrDict(self.params))

    def forward(self, x: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        outputs = self.model(x["mel_spec"])
        return {"slice": outputs}
