from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn

from loss.components.loss_modules import LOSS_MODULES


@dataclass
class DecoderCrossEntropy:
    """
    Standard cross-entropy-loss for training a decoder transformer for sequence generation.
    """

    name: str
    weight: float
    base_loss: nn.Module

    def __call__(
        self, estimation: dict[str, torch.Tensor], target: dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """
        Forward method for this configuration

        Args:
            est (Dict[str, torch.Tensor]): Dictionary expecting BS x V x cls in key "logits"
            ref (Dict[str, torch.Tensor]): Dictionary expecting BS x V in key "latent indices"

        Returns:
            torch.Tensor: Loss
        """
        logits = estimation["logits"][:-1]
        target_indices = target["latent indices"][1:]
        return self.base_loss(logits.transpose(1, 2), target_indices.long())


def build_lm_cross_entropy_from_cfg(
    name: str, loss_cfg: dict[str, Any]
) -> DecoderCrossEntropy:
    loss_module = LOSS_MODULES[loss_cfg.get("base_loss", "ce")]
    return DecoderCrossEntropy(name, loss_cfg.get("weight", 1.0), loss_module)
