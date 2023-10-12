from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn

from loss.components.loss_modules import LOSS_MODULES


@dataclass
class AlignLoss:
    """
    VQ-VAE codebook alignment loss
    """

    name: str
    weight: float
    base_loss: nn.Module

    def __call__(
        self, estimation: dict[str, torch.Tensor], target: dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """
        Computes the VQ-VAE codebook alignment loss

        Args:
            estimation (dict[str, torch.Tensor]): Network estimation
            target (dict[str, torch.Tensor]): Ground truth reference

        Returns:
            torch.Tensor: Loss
        """
        emb = estimation["emb"]
        z_e = target["z_e"]

        return self.base_loss(emb, z_e.detach())


def build_alignment_loss_from_cfg(name: str, loss_cfg: dict[str, Any]) -> AlignLoss:
    """
    Builds an alignment loss object

    Args:
        name (str): loss name
        loss_cfg (dict[str, Any]): loss config

    Returns:
        AlignLoss: Alignment loss object
    """
    loss_module = LOSS_MODULES[loss_cfg.get("base_loss", "mse")]
    return AlignLoss(name, loss_cfg.get("weight", 1.0), loss_module)


@dataclass
class CommitLoss:
    """
    VQ-VAE codebook commitment loss
    """

    name: str
    weight: float
    base_loss: nn.Module

    def __call__(
        self, estimation: dict[str, torch.Tensor], target: dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """
        Computes the VQ-VAE codebook commitment loss

        Args:
            estimation (dict[str, torch.Tensor]): Network estimation
            target (dict[str, torch.Tensor]): Ground truth reference

        Returns:
            torch.Tensor: Loss
        """
        emb = estimation["emb"]
        z_e = target["z_e"]

        return self.base_loss(emb.detach(), z_e)


def build_commitment_loss_from_cfg(name: str, loss_cfg: dict[str, Any]) -> CommitLoss:
    """
    Builds an commitment loss object

    Args:
        name (str): loss name
        loss_cfg (dict[str, Any]): loss config

    Returns:
        AlignLoss: Commitment loss object
    """
    loss_module = LOSS_MODULES[loss_cfg.get("base_loss", "mse")]
    return CommitLoss(name, loss_cfg.get("weight", 1.0), loss_module)
