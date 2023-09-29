from abc import abstractmethod
from typing import Any, TYPE_CHECKING, Optional, Protocol

import pytorch_lightning as pl
import torch
import torch.nn as nn

from utils.containers import (
    LearningParameters,
    DiffusionParameters,
)

if TYPE_CHECKING:
    from loss.aggregators import LossAggregator


class BaseLightningModule(pl.LightningModule):
    def __init__(
        self,
        base_model: nn.Module,
        learning_params: LearningParameters,
        loss_aggregator: Optional["LossAggregator"] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Any = None,
        **kwargs,
    ) -> None:
        super().__init__()

        # Initialize variables
        self.learning_params = learning_params
        self.model = base_model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.loss_aggregator = loss_aggregator

    @abstractmethod
    def forward(self, x: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        ...

    def _configure_scheduler_settings(
        self, interval: str, monitor: str, frequency: int
    ):
        if self.scheduler is None:
            raise TypeError("Must include a scheduler")
        return {
            "scheduler": self.scheduler,
            "interval": interval,
            "monitor": monitor,
            "frequency": frequency,
        }

    def configure_optimizers(self):
        print(self.scheduler)
        if self.scheduler is None:
            return [self.optimizer]
        else:
            scheduler_settings = self._configure_scheduler_settings(
                self.learning_params.interval,
                self.learning_params.loss_monitor,
                self.learning_params.frequency,
            )
            return [self.optimizer], [scheduler_settings]


class DiffusionScheduler(Protocol):
    betas: torch.Tensor
    alphas: torch.Tensor
    alphas_cumprod: torch.Tensor
    alphas_cumprod_prev: torch.Tensor
    sqrt_recip_alphas: torch.Tensor
    sqrt_alphas_cumprod: torch.Tensor
    sqrt_one_minus_alphas_cumprod: torch.Tensor
    posterior_variance: torch.Tensor

    def __init__(self, num_steps: int, device: str = "cpu") -> None:
        ...


class BaseDiffusionModel(BaseLightningModule):
    def __init__(
        self,
        base_model: nn.Module,
        learning_params: LearningParameters,
        diffusion_params: DiffusionParameters,
        loss_aggregator: Optional["LossAggregator"] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Any = None,
        **kwargs,
    ) -> None:
        super().__init__(
            base_model, learning_params, loss_aggregator, optimizer, scheduler, **kwargs
        )
        self.diffusion_params = diffusion_params

    @abstractmethod
    def forward(
        self, x: dict[str, torch.Tensor], t: torch.Tensor, cond: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        ...

    @abstractmethod
    def get_loss(self, x_0: dict[str, torch.Tensor], t: torch.Tensor) -> torch.Tensor:
        ...

    @abstractmethod
    def sample_timestep(
        self,
        x: dict[str, torch.Tensor],
        t: torch.Tensor,
        cond: dict[str, torch.Tensor] = {},
        verbose: bool = False,
    ) -> dict[str, torch.Tensor]:
        ...

    @abstractmethod
    def denoise(
        self,
        noisy_input: dict[str, torch.Tensor],
        cond: dict[str, torch.Tensor] = {},
        show_process_plots: bool = False,
    ) -> dict[str, torch.Tensor]:
        ...
