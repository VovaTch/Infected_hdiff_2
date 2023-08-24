from typing import Dict

import torch

from .base import LossComponent


class AlignLoss(LossComponent):
    """
    VQ-VAE codebook alignment loss
    """

    def __call__(
        self,
        est: Dict[str, torch.Tensor],
        ref: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        emb = est["emb"]
        z_e = ref["z_e"]

        return self.loss_module(emb, z_e.detach())


class CommitLoss(LossComponent):
    """
    VQ-VAE codebook commitment loss
    """

    def __call__(
        self,
        est: Dict[str, torch.Tensor],
        ref: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        emb = est["emb"]
        z_e = ref["z_e"]

        return self.loss_module(emb.detach(), z_e)
