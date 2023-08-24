from typing import Any, Dict

import torch
import torch.nn as nn
from torchvision.ops import sigmoid_focal_loss

from loss.base import LossAggregator, LossComponent, MelSpecLossComponent
from .mel_spec_loss import MelSpecLoss
from .rec_loss import RecLoss, NoisePredLoss
from .codebook_losses import AlignLoss, CommitLoss
from .lm_cross_entropy_loss import DecoderCrossEntropy
from .loss_aggregator import WeightedSumAggregator
from .base import ILossAggregatorFactory, LossAggregator
from models import MelSpecConverterFactory
from utils.containers import MelSpecParameters


LOSS_MODULES: Dict[str, nn.Module] = {
    "mse": nn.MSELoss(),
    "l1": nn.L1Loss(),
    "huber": nn.SmoothL1Loss(),
    "bce": nn.BCEWithLogitsLoss(),
    "ce": nn.CrossEntropyLoss(),
    "focal": sigmoid_focal_loss,
}

LOSS_COMPONENTS: Dict[str, nn.Module] = {
    "melspec": MelSpecLoss,
    "reconstruction": RecLoss,
    "alignment": AlignLoss,
    "commitment": CommitLoss,
    "noise": NoisePredLoss,
    "decoderCE": DecoderCrossEntropy,
}

LOSS_AGGREGATORS: Dict[str, WeightedSumAggregator] = {
    "weighted_sum": WeightedSumAggregator,
}


class LossAggregatorFactory(ILossAggregatorFactory):
    @staticmethod
    def build_loss_component(
        name: str, component_config: Dict[str, Any]
    ) -> LossComponent:
        component_type = component_config["type"]
        base_loss_type = component_config["base_loss_type"]
        base_loss = LOSS_MODULES[base_loss_type]
        loss_component = LOSS_COMPONENTS[component_type](
            name, component_config["weight"], base_loss, **component_config
        )
        return loss_component

    @staticmethod
    def build_melspec_loss_component(
        name: str, component_config: Dict[str, Any]
    ) -> MelSpecLossComponent:
        component_type = component_config["type"]
        base_loss_type = component_config["base_loss_type"]
        mel_spec_params = MelSpecParameters(**component_config["melspec_params"])
        mel_spec_converter = MelSpecConverterFactory.build_mel_spec_converter(
            "simple", mel_spec_params
        )
        base_loss = LOSS_MODULES[base_loss_type]
        loss_component = LOSS_COMPONENTS[component_type](
            name,
            component_config["weight"],
            base_loss,
            mel_spec_converter,
            **component_config
        )
        return loss_component

    @staticmethod
    def build_loss_aggregator(loss_config: Dict[str, Any]) -> LossAggregator:
        loss_components = []
        for loss_component_name in loss_config:
            if loss_component_name == "aggregator_type":
                loss_aggregator_type = loss_config["aggregator_type"]
                continue
            if loss_config[loss_component_name]["type"] == "melspec":
                loss_component = LossAggregatorFactory.build_melspec_loss_component(
                    loss_component_name, loss_config[loss_component_name]
                )
            else:
                loss_component = LossAggregatorFactory.build_loss_component(
                    loss_component_name, loss_config[loss_component_name]
                )
            loss_components.append(loss_component)
        loss_aggregator = LOSS_AGGREGATORS[loss_aggregator_type](
            loss_components, **loss_config
        )
        return loss_aggregator
