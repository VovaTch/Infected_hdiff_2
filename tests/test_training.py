import pytest
from loaders.build import build_mel_spec_module

from loss.factory import build_loss_aggregator
from models.build import build_diffwave_diffusion_vocoder, build_mel_spec_converter
from models.mel_spec_converters import MelSpecConverter
from models.diffwave_vocoder import VocoderDiffusionModel
from utils.containers import (
    LearningParameters,
    MelSpecParameters,
    MusicDatasetParameters,
    parse_cfg_for_vocoder,
)
from utils.others import load_config
from utils.trainer import initialize_trainer


@pytest.fixture
def vocoder_diffusion_model() -> VocoderDiffusionModel:
    cfg = load_config("config/test_config.yaml")
    cfg["loss"] = {
        "noise": {"type": "noise", "weight": 1.0, "base_loss_type": "mse"},
        "aggregator_type": "weighted_sum",
    }
    loss_aggregator = build_loss_aggregator(cfg)
    return build_diffwave_diffusion_vocoder(cfg, loss_aggregator=loss_aggregator)


@pytest.fixture
def parameter_containers() -> (
    tuple[LearningParameters, MusicDatasetParameters, MelSpecParameters]
):
    cfg = load_config("config/test_config.yaml")
    return parse_cfg_for_vocoder(cfg)


def test_short_training(
    vocoder_diffusion_model,
    parameter_containers: tuple[
        LearningParameters, MusicDatasetParameters, MelSpecParameters
    ],
):
    learning_param, dataset_params, mel_spec_params = parameter_containers
    learning_param.epochs = 1
    data_module = build_mel_spec_module(dataset_params, learning_param, mel_spec_params)
    trainer = initialize_trainer(learning_param)
    trainer.fit(vocoder_diffusion_model, datamodule=data_module)  # type: ignore
