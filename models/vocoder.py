from typing import Any, Dict

from pytorch_lightning.utilities.types import STEP_OUTPUT
import torch

from .base import BaseLightningModule


class Vocoder(BaseLightningModule):
    def forward(self, x: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        assert (
            "mel_spec" in x
        ), "Input dictionary must include 'mel_spec' key that represents a mel spectrogram image."
        slice_input = x["mel_spec"]
        output = self.model(slice_input)
        return output

    def training_step(self, batch: Dict[str, Any], batch_idx: int) -> STEP_OUTPUT:
        output = self.forward(batch)
        loss_dict = self.loss_aggregator(output, batch)
        for key, value in loss_dict.items():
            prog_bar = True if "total_loss" in key else False
            self.log()
