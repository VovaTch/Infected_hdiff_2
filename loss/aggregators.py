from dataclasses import dataclass
from typing import Protocol

import torch

from .components.generic import LossComponent


@dataclass
class LossOutput:
    total: torch.Tensor
    individuals: dict[str, torch.Tensor]


@dataclass
class LossAggregator(Protocol):
    components: list[LossComponent]

    def compute(
        self,
        estimation: dict[str, torch.Tensor],
        target: dict[str, torch.Tensor],
    ) -> LossOutput:
        """Call method for loss aggregation

        Args:
            estimation (dict[str, torch.Tensor]): Network estimation dictionary
            target (dict[str, torch.Tensor]): Target dictionary

        Returns:
            LossOutput: LossOutput object representing the total loss and the individual parts
        """
        ...


@dataclass
class WeightedSumAggregator:
    components: list[LossComponent]

    def compute(
        self, estimation: dict[str, torch.Tensor], target: dict[str, torch.Tensor]
    ) -> LossOutput:
        loss = LossOutput(torch.tensor(0.0), {})

        for component in self.components:
            ind_loss = component(estimation, target)
            loss.total = loss.total.to(ind_loss.device)
            loss.total += component.weight * ind_loss
            loss.individuals[component.name] = ind_loss

        return loss


AGGREGATORS: dict[str, type[LossAggregator]] = {"weighted_sum": WeightedSumAggregator}


def register_aggregator(new_aggregator: type[LossAggregator], type: str) -> None:
    AGGREGATORS[type] = new_aggregator
