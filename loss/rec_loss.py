from typing import Any, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import LossComponent


class RecLoss(LossComponent):
    def __init__(
        self, loss_name: str, loss_weight: float, loss_module: nn.Module, **kwargs
    ) -> None:
        super().__init__(loss_name, loss_weight, loss_module, **kwargs)
        self.use_tanh = False
        self.phase_parameter = 1

        if "use_tanh" in kwargs:
            if kwargs:
                self.use_tanh = True

        if "phase_parameter" in kwargs:
            self.phase_parameter = kwargs["phase_parameter"]

    def __call__(self, est: Dict[str, Any], ref: Dict[str, Any]) -> torch.Tensor:
        pred_slice = est["slice"]
        target_slice = ref["slice"]

        if self.use_tanh:
            return self._phased_loss(
                F.tanh(pred_slice),
                F.tanh(target_slice),
                phase_parameter=self.phase_parameter,
            )
        else:
            return self._phased_loss(
                pred_slice, target_slice, phase_parameter=self.phase_parameter
            )

    def _phased_loss(
        self, est: torch.Tensor, ref: torch.Tensor, phase_parameter: int = 10
    ):
        loss_vector = torch.zeros(phase_parameter * 2).to(est.device)
        for idx in range(phase_parameter):
            if idx == 0:
                loss_vector[idx * 2] = self.loss_module(est, ref)
                loss_vector[idx * 2 + 1] = loss_vector[idx * 2] + 1e-6
            else:
                loss_vector[idx * 2] = self.loss_module(
                    est[:, :, idx:], ref[:, :, :-idx]
                )
                loss_vector[idx * 2 + 1] = self.loss_module(
                    est[:, :, :-idx], ref[:, :, idx:]
                )
        return loss_vector.min()


class NoisePredLoss(LossComponent):
    """
    Basic loss for reconstructing noise, used in diffusion
    """

    def __call__(self, est: Dict[str, Any], ref: Dict[str, Any]) -> torch.Tensor:
        noise = ref["noise"]
        noise_pred = est["noise_pred"]

        return self.loss_module(noise, noise_pred)
