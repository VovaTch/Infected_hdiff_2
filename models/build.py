from typing import Any, Optional, TYPE_CHECKING

from utils.containers import (
    DiffusionParameters,
    LearningParameters,
    MelSpecParameters,
    MusicDatasetParameters,
    Res1DDecoderParameters,
)
from .mel_spec_converters import MelSpecConverter, MEL_SPEC_CONVERTERS
from .diffwave_vocoder import VocoderDiffusionModel, DiffwaveWrapper
from .res1d_decoder import Decoder1D, Res1DLightningModule
from .diffusion_schedulers import DIFFUSION_SCHEDULERS

if TYPE_CHECKING:
    from loss.aggregators import LossAggregator


def build_mel_spec_converter(
    type: str, mel_spec_params: MelSpecParameters
) -> MelSpecConverter:
    if type not in MEL_SPEC_CONVERTERS:
        raise ValueError(f"Unknown converter type {type}")
    mel_spec_converter = MEL_SPEC_CONVERTERS[type](mel_spec_params)
    return mel_spec_converter


def build_diffwave_diffusion_vocoder(
    cfg: dict[str, Any],
    loss_aggregator: Optional["LossAggregator"] = None,
    weights_path: str | None = None,
) -> VocoderDiffusionModel:
    # Build diffwave module
    learning_params = LearningParameters(**cfg["learning"])
    mel_spec_params = MelSpecParameters(**cfg["image_mel_spec_params"])
    dataset_params = MusicDatasetParameters(**cfg["dataset"])
    diffwave_module = DiffwaveWrapper(learning_params, mel_spec_params, dataset_params)

    # Build diffwave wrapper
    device = "cuda" if learning_params.num_devices > 0 else "cpu"
    num_steps = cfg["diffusion"]["num_steps"]
    scheduler = DIFFUSION_SCHEDULERS[cfg["diffusion"]["scheduler"]](num_steps, device)
    diffusion_params = DiffusionParameters(scheduler, num_steps)
    if not weights_path:
        return VocoderDiffusionModel(
            diffwave_module,
            learning_params,
            diffusion_params,
            loss_aggregator=loss_aggregator,
            scheduler=None,
        )
    else:
        return VocoderDiffusionModel.load_from_checkpoint(
            weights_path,
            base_model=diffwave_module,
            learning_params=learning_params,
            diffusion_params=diffusion_params,
            loss_aggregator=loss_aggregator,
            scheduler=None,
        )


def build_res1d_vocoder(
    cfg: dict[str, Any],
    loss_aggregator: Optional["LossAggregator"] = None,
    weights_path: str | None = None,
) -> Res1DLightningModule:
    learning_params = LearningParameters(**cfg["learning"])
    mel_spec_params = MelSpecParameters(**cfg["image_mel_spec_params"])
    dataset_params = MusicDatasetParameters(**cfg["dataset"])
    model_params = Res1DDecoderParameters(**cfg["model"])
    base_model = Decoder1D(model_params, mel_spec_params, dataset_params.slice_length)

    # Build the decoder
    if not weights_path:
        return Res1DLightningModule(
            base_model=base_model,
            learning_params=learning_params,
            loss_aggregator=loss_aggregator,
        )
    else:
        return Res1DLightningModule.load_from_checkpoint(
            weights_path,
            base_model=base_model,
            learning_params=learning_params,
            loss_aggregator=loss_aggregator,
        )
