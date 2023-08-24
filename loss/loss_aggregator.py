from typing import Any, Dict

import torch

from .base import LossAggregator


class WeightedSumAggregator(LossAggregator):
    def __call__(
        self, est: Dict[str, Any], ref: Dict[str, Any]
    ) -> Dict[str, torch.Tensor]:
        losses = {"total_loss": torch.tensor(0.0)}
        for loss_component in self.loss_components:
            losses[
                loss_component.loss_name
            ] = loss_component.loss_weight * loss_component(est, ref)
            losses["total_loss"] += losses[loss_component.loss_name]

        return losses
