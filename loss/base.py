from abc import ABC, abstractmethod
from typing import Dict, Any, List

import torch
import torch.nn as nn

from models import MelSpecConverter


class LossComponent(ABC):
    loss_module: nn.Module
    loss_name: str
    loss_weight: float

    def __init__(
        self, loss_name: str, loss_weight: float, loss_module: nn.Module, **kwargs
    ) -> None:
        self.loss_name = loss_name
        self.loss_weight = loss_weight
        self.loss_module = loss_module

    @abstractmethod
    def __call__(self, est: Dict[str, Any], ref: Dict[str, Any]) -> torch.Tensor:
        ...


class MelSpecLossComponent(LossComponent):
    def __init__(
        self,
        loss_name: str,
        loss_weight: float,
        loss_module: nn.Module,
        mel_spec_converter: MelSpecConverter,
        **kwargs
    ) -> None:
        super().__init__(loss_name, loss_weight, loss_module)
        self.mel_spec_converter = mel_spec_converter


class LossAggregator(ABC):
    loss_components: List[LossComponent]

    def __init__(self, loss_components: List[LossComponent], **kwargs) -> None:
        super().__init__()
        self.loss_components = loss_components

    @abstractmethod
    def __call__(
        self, est: Dict[str, Any], ref: Dict[str, Any]
    ) -> Dict[str, torch.Tensor]:
        ...


class ILossAggregatorFactory(ABC):
    @staticmethod
    @abstractmethod
    def build_loss_component(
        name: str, component_config: Dict[str, Any]
    ) -> LossComponent:
        ...

    @staticmethod
    @abstractmethod
    def build_melspec_loss_component(
        name: str, component_config: Dict[str, Any]
    ) -> MelSpecLossComponent:
        ...

    @staticmethod
    @abstractmethod
    def build_loss_aggregator(loss_config: Dict[str, Any]) -> LossAggregator:
        ...
