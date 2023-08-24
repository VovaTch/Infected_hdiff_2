from typing import Dict

import torch
import torch.nn as nn

from models import MelSpecConverter
from .base import MelSpecLossComponent


class MelSpecLoss(MelSpecLossComponent):
    """
    This loss is a reconstruction loss of a mel spectrogram, convert the inputs into a spectrogram and
    compute reconstruction loss
    """

    def __init__(
        self,
        loss_name: str,
        loss_weight: float,
        loss_module: nn.Module,
        mel_spec_converter: MelSpecConverter,
        **kwargs,
    ) -> None:
        super().__init__(
            loss_name, loss_weight, loss_module, mel_spec_converter, **kwargs
        )
        if "lin_start" in kwargs:
            self.lin_start = kwargs["lin_start"]
        else:
            self.lin_start = 1.0

        if "lin_end" in kwargs:
            self.lin_end = kwargs["lin_end"]
        else:
            self.lin_end = 1.0

    def _mel_spec_and_process(self, x: torch.Tensor):
        """
        To prepare the mel spectrogram loss, everything needs to be prepared.

        Args:
            x (torch.Tensor): Input, will be flattened
        """
        lin_vector = torch.linspace(
            self.lin_start,
            self.lin_end,
            self.mel_spec_converter.mel_spec.n_mels,
        )
        eye_mat = torch.diag(lin_vector).to(x.device)
        mel_out = self.mel_spec_converter.convert(x.flatten(start_dim=0, end_dim=1))
        mel_out = torch.tanh(eye_mat @ mel_out)
        return mel_out

    def __call__(
        self, est: Dict[str, torch.Tensor], ref: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        pred_slice = est["slice"]
        target_slice = ref["slice"]

        self.mel_spec_converter.mel_spec = self.mel_spec_converter.mel_spec.to(
            pred_slice.device
        )

        return self.loss_module(
            self._mel_spec_and_process(pred_slice),
            self._mel_spec_and_process(target_slice),
        )
