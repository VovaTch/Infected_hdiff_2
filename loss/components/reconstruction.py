from dataclasses import dataclass
from typing import Any, Callable
import torch
import torch.nn as nn

from .loss_modules import LOSS_MODULES
from utils.transform_funcs import TRANSFORM_FUNCS


@dataclass
class RecLoss:
    """Reconstruction loss for sound-waves"""

    name: str
    weight: float
    base_loss: nn.Module

    # Loss-specific parameters
    transform_func: Callable[[torch.Tensor], torch.Tensor] = lambda x: x
    phase_parameter: int = 1

    def __call__(
        self, estimation: dict[str, torch.Tensor], target: dict[str, torch.Tensor]
    ) -> torch.Tensor:
        pred_slice = estimation["slice"]
        target_slice = target["slice"]

        return self._phased_loss(
            self.transform_func(pred_slice),
            self.transform_func(target_slice),
            phase_parameter=self.phase_parameter,
        )

    def _phased_loss(
        self, estimation: torch.Tensor, target: torch.Tensor, phase_parameter: int = 10
    ):
        loss_vector = torch.zeros(phase_parameter * 2).to(estimation.device)
        for idx in range(phase_parameter):
            if idx == 0:
                loss_vector[idx * 2] = self.base_loss(estimation, target)
                loss_vector[idx * 2 + 1] = loss_vector[idx * 2] + 1e-6
                continue

            loss_vector[idx * 2] = self.base_loss(
                estimation[:, :, idx:], target[:, :, :-idx]
            )
            loss_vector[idx * 2 + 1] = self.base_loss(
                estimation[:, :, :-idx], target[:, :, idx:]
            )

        return loss_vector.min()


def build_rec_loss_from_cfg(name: str, loss_cfg: dict[str, Any]) -> RecLoss:
    loss_module = LOSS_MODULES[loss_cfg.get("base_loss", "mse")]
    transform_func = TRANSFORM_FUNCS[loss_cfg.get("transform_func", "none")]
    phase_parameter = loss_cfg.get("phase_parameter", 1)
    return RecLoss(
        name,
        loss_cfg.get("weight", 1.0),
        loss_module,
        transform_func=transform_func,
        phase_parameter=phase_parameter,
    )


@dataclass
class NoisePredLoss:
    """
    Basic loss for reconstructing noise, used in diffusion
    """

    name: str
    weight: float
    base_loss: nn.Module

    def __call__(
        self, estimation: dict[str, torch.Tensor], target: dict[str, torch.Tensor]
    ) -> torch.Tensor:
        noise = target["noise"]
        noise_pred = estimation["noise_pred"]

        return self.base_loss(noise, noise_pred)


def build_noise_pred_loss_from_cfg(
    name: str, loss_cfg: dict[str, Any]
) -> NoisePredLoss:
    loss_module = LOSS_MODULES[loss_cfg.get("base_loss", "mse")]
    return NoisePredLoss(name, loss_cfg.get("weight", 1.0), loss_module)
