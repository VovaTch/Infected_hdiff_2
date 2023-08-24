from typing import Any, Dict

import torch
import torch.nn as nn

from .base import LossComponent


class DecoderCrossEntropy(LossComponent):
    """
    Standard cross-entropy-loss for training a decoder transformer for sequence generation.
    """

    def __call__(
        self, est: Dict[str, torch.Tensor], ref: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """
        Forward method for this configuration

        Args:
            est (Dict[str, torch.Tensor]): Dictionary expecting BS x V x cls in key "logits"
            ref (Dict[str, torch.Tensor]): Dictionary expecting BS x V in key "latent indices"

        Returns:
            torch.Tensor: Loss
        """
        logits = est["logits"][:-1]
        target_indices = ref["latent indices"][1:]
        return self.loss_module(logits.transpose(1, 2), target_indices.long())
