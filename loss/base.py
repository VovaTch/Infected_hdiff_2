from abc import ABC, abstractmethod
from typing import Dict, Any, List

import torch
import torch.nn as nn
from torchvision.ops import sigmoid_focal_loss


LOSS_MODULES: Dict[str, nn.Module] = {
    "mse": nn.MSELoss(),
    "l1": nn.L1Loss(),
    "huber": nn.SmoothL1Loss(),
    "bce": nn.BCEWithLogitsLoss(),
    "ce": nn.CrossEntropyLoss(),
    "focal": sigmoid_focal_loss,
}


class LossComponent(ABC):
    loss_module: nn.Module
    loss_name: str
    loss_weight: float

    def __init__(
        self, loss_name: str, loss_weight: float, loss_module: nn.Module
    ) -> None:
        self.loss_name = loss_name
        self.loss_weight = loss_weight
        self.loss_module = loss_module

    @abstractmethod
    def __call__(self, est: Dict[str, Any], ref: Dict[str, Any]) -> torch.Tensor:
        ...


class LossAggregator(ABC):
    loss_components: List[LossComponent]

    def __init__(self, loss_components: List[LossComponent]) -> None:
        super().__init__()
        self.loss_components = loss_components

    @abstractmethod
    def __call__(
        self, est: Dict[str, Any], ref: Dict[str, Any]
    ) -> Dict[str, torch.Tensor]:
        ...
