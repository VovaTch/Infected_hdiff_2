from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Any

import pytorch_lightning as pl
import torch
import torch.nn as nn

from utils.containers import LearningParameters


class BaseLightningModule(pl.LightningModule):
    def __init__(
        self,
        base_model: nn.Module,
        learning_params: LearningParameters,
        loss_aggregator=None,
        optimizer: torch.optim.Optimizer = None,
        scheduler: torch.optim.lr_scheduler = None,
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
    def forward(self, x: Dict[str, Any]) -> object:
        ...

    def _configure_scheduler_settings(
        self, interval: str, monitor: str, frequency: int
    ):
        assert self.scheduler is not None, "Must include a scheduler"
        return {
            "scheduler": self.scheduler,
            "interval": interval,
            "monitor": monitor,
            "frequency": frequency,
        }

    def configure_optimizers(self):
        if self.scheduler is None:
            return [self.optimizer]
        else:
            scheduler_settings = self._configure_scheduler_settings(
                self.learning_params.interval,
                self.learning_params.loss_monitor,
                self.learning_params.frequency,
            )
            return [self.optimizer], [scheduler_settings]
