from typing import Any, Callable, Protocol
from dataclasses import dataclass

import torch
import torch.nn as nn

from .codebook import (
    AlignLoss,
    CommitLoss,
    build_alignment_loss_from_cfg,
    build_commitment_loss_from_cfg,
)
from .lm_cross_entropy import DecoderCrossEntropy, build_lm_cross_entropy_from_cfg
from .mel_spec import (
    MelSpecLoss,
    build_mel_spec_loss_from_cfg,
    MelSpecDiffusionLoss,
    build_mel_spec_diff_loss_from_cfg,
)
from .reconstruction import (
    RecLoss,
    NoisePredLoss,
    DiffReconstructionLoss,
    build_rec_loss_from_cfg,
    build_noise_pred_loss_from_cfg,
    build_diff_rec_loss_from_cfg,
)


@dataclass
class LossComponent(Protocol):
    """
    Loss component object protocol

    Fields:
        name (str): loss name
        weight (float): loss relative weight for computation (e.g. weighted sum)
        base_loss (nn.Module): base loss module specific for the loss
    """

    name: str
    weight: float
    base_loss: nn.Module

    def __call__(
        self, estimation: dict[str, torch.Tensor], target: dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """
        Call method for outputting the loss

        Args:
            estimation (dict[str, torch.Tensor]): Network estimation
            target (dict[str, torch.Tensor]): Ground truth reference

        Returns:
            torch.Tensor: loss
        """
        ...


COMPONENTS: dict[str, Any] = {
    "align": AlignLoss,
    "commit": CommitLoss,
    "lm_ce": DecoderCrossEntropy,
    "melspec": MelSpecLoss,
    "melspecdiff": MelSpecDiffusionLoss,
    "rec": RecLoss,
    "noise": NoisePredLoss,
    "diff_rec": DiffReconstructionLoss,
}

COMPONENT_FACTORIES: dict[str, Callable[[str, dict[str, Any]], Any]] = {
    "align": build_alignment_loss_from_cfg,
    "commit": build_commitment_loss_from_cfg,
    "lm_ce": build_lm_cross_entropy_from_cfg,
    "melspec": build_mel_spec_loss_from_cfg,
    "melspecdiff": build_mel_spec_diff_loss_from_cfg,
    "rec": build_rec_loss_from_cfg,
    "noise": build_noise_pred_loss_from_cfg,
    "diff_rec": build_diff_rec_loss_from_cfg,
}


def register_component(
    new_component: LossComponent,
    new_component_factory: Callable[[str, dict[str, Any]], Any],
    type: str,
) -> None:
    """
    Registers a loss component with a factory method into the component and factory dictionaries

    Args:
        new_component (LossComponent): Loss component
        new_component_factory (Callable[[str, dict[str, Any]], Any]): Loss component factory function/method
        type (str): String describing the type, used to call the loss
    """
    COMPONENTS[type] = new_component
    COMPONENT_FACTORIES[type] = new_component_factory
