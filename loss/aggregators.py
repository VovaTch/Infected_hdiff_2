from dataclasses import dataclass
from typing import Protocol

import torch

from .components.generic import LossComponent


@dataclass
class LossOutput:
    """
    Loss output object that contains individual components named, and the total loss from the aggregator.

    Fields:
        *   total (Tensor): Total loss from the aggregator
        *   individuals (dict[str, Tensor]): Individual loss component values.
    """

    total: torch.Tensor
    individuals: dict[str, torch.Tensor]


@dataclass
class LossAggregator(Protocol):
    """
    Loss aggregator protocol, uses a math operation on component losses to compute a total loss. For example, weighted sum.
    """

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
    """
    Weighted sum loss component
    """

    components: list[LossComponent]

    def compute(
        self, estimation: dict[str, torch.Tensor], target: dict[str, torch.Tensor]
    ) -> LossOutput:
        """
        Forward method to compute the weighted sum

        Args:
            estimation (dict[str, torch.Tensor]): Network estimation
            target (dict[str, torch.Tensor]): Ground truth reference

        Returns:
            LossOutput: Loss output object with total loss and individual losses
        """
        loss = LossOutput(torch.tensor(0.0), {})

        for component in self.components:
            ind_loss = component(estimation, target)
            loss.total = loss.total.to(ind_loss.device)
            loss.total += component.weight * ind_loss
            loss.individuals[component.name] = ind_loss

        return loss


AGGREGATORS: dict[str, type[LossAggregator]] = {"weighted_sum": WeightedSumAggregator}


def register_aggregator(new_aggregator: type[LossAggregator], type: str) -> None:
    """
    Registers a new aggregator into the aggregator dict

    Args:
        new_aggregator (type[LossAggregator]): New loss aggregator
        type (str): Aggregator name
    """
    AGGREGATORS[type] = new_aggregator
