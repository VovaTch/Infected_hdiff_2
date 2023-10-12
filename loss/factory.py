from typing import Any

from .aggregators import LossAggregator, AGGREGATORS
from .components.generic import COMPONENT_FACTORIES


def build_loss_aggregator(cfg: dict[str, Any]) -> LossAggregator:
    """
    Builds a loss aggregator object

    Args:
        cfg (dict[str, Any]): Configuration dictionary

    Returns:
        LossAggregator: Loss aggregation object
    """
    aggregator_type = cfg["loss"]["aggregator_type"]
    components = []
    for component_key in cfg["loss"]:
        if component_key == "aggregator_type":
            continue
        component_cfg = cfg["loss"][component_key]
        component_type = component_cfg["type"]
        component = COMPONENT_FACTORIES[component_type](component_key, component_cfg)
        components.append(component)

    return AGGREGATORS[aggregator_type](components)
