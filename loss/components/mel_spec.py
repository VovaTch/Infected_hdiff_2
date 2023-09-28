from dataclasses import dataclass
from typing import Any, Callable, TYPE_CHECKING

import torch
import torch.nn as nn
from loss.components.loss_modules import LOSS_MODULES

from utils.containers import MelSpecParameters
from models.build import build_mel_spec_converter
from utils.transform_funcs import TRANSFORM_FUNCS

if TYPE_CHECKING:
    from models.mel_spec_converters import MelSpecConverter


@dataclass
class MelSpecLoss:
    """
    This loss is a reconstruction loss of a mel spectrogram, convert the inputs into a spectrogram and
    compute reconstruction loss
    """

    name: str
    weight: float
    base_loss: nn.Module

    # Loss-specific parameters
    mel_spec_converter: "MelSpecConverter"
    transform_func: Callable[[torch.Tensor], torch.Tensor] = lambda x: torch.tanh(x)
    lin_start: float = 1.0
    lin_end: float = 1.0

    def _mel_spec_and_process(self, x: torch.Tensor) -> torch.Tensor:
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
        mel_out = self.transform_func(eye_mat @ mel_out)
        return mel_out

    def __call__(
        self, estimation: dict[str, torch.Tensor], target: dict[str, torch.Tensor]
    ) -> torch.Tensor:
        pred_slice = estimation["slice"]
        target_slice = target["slice"]

        self.mel_spec_converter.mel_spec = self.mel_spec_converter.mel_spec.to(
            pred_slice.device
        )

        return self.base_loss(
            self._mel_spec_and_process(pred_slice),
            self._mel_spec_and_process(target_slice),
        )


def build_mel_spec_loss_from_cfg(name: str, loss_cfg: dict[str, Any]) -> MelSpecLoss:
    # Create mel spec converter
    mel_spec_params = MelSpecParameters(**loss_cfg["melspec_params"])
    mel_spec_converter = build_mel_spec_converter(
        type="simple", mel_spec_params=mel_spec_params
    )
    loss_module = LOSS_MODULES[loss_cfg.get("base_loss", "mse")]
    transform_func = TRANSFORM_FUNCS[loss_cfg.get("transform_func", "tanh")]
    lin_start = loss_cfg.get("lin_start", 1.0)
    lin_end = loss_cfg.get("lin_end", 1.0)

    # Create mel-spec loss
    return MelSpecLoss(
        name,
        loss_cfg.get("weight", 1.0),
        loss_module,
        mel_spec_converter,
        transform_func=transform_func,
        lin_start=lin_start,
        lin_end=lin_end,
    )
