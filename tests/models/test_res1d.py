import pytest
import torch
import loss
from loss.factory import build_loss_aggregator

from models.build import build_mel_spec_converter, build_res1d_vocoder
from models.mel_spec_converters import MelSpecConverter
from models.res1d_decoder import Res1DLightningModule
from utils.containers import MelSpecParameters
from utils.others import load_config


@pytest.fixture
def res1d_vocoder() -> Res1DLightningModule:
    cfg = load_config("config/test_config.yaml")
    cfg["loss"] = {
        "aggregator_type": "weighted_sum",
        "reconstruction_loss": {
            "type": "rec",
            "weight": 1.0,
            "base_loss_type": "mse",
            "phase_parameter": 1,
        },
    }
    cfg["image_mel_spec_params"] = cfg["image_mel_spec_params_res1d"]
    loss_aggregator = build_loss_aggregator(cfg)
    return build_res1d_vocoder(cfg, loss_aggregator=loss_aggregator)


@pytest.fixture
def mel_spec_converter() -> MelSpecConverter:
    cfg = load_config("config/test_config.yaml")
    mel_spec_params = MelSpecParameters(**cfg["image_mel_spec_params_res1d"])
    return build_mel_spec_converter("simple", mel_spec_params)


def test_forward(
    res1d_vocoder: Res1DLightningModule, mel_spec_converter: MelSpecConverter
):
    batch_size = 3
    test_slice = torch.randn((batch_size, 1, 32768))
    test_mel_spec = mel_spec_converter.convert(test_slice).squeeze(1)
    inputs = {"slice": test_slice, "mel_spec": test_mel_spec}
    outputs = res1d_vocoder.forward(inputs)
    assert outputs["slice"].shape == inputs["slice"].shape


def test_training_step(
    res1d_vocoder: Res1DLightningModule, mel_spec_converter: MelSpecConverter
):
    batch_size = 3
    res1d_vocoder = res1d_vocoder.to("cuda")
    mel_spec_converter.mel_spec = mel_spec_converter.mel_spec.to("cuda")
    test_slice = torch.randn((batch_size, 1, 32768)).to("cuda")
    test_mel_spec = mel_spec_converter.convert(test_slice)
    batch = {"slice": test_slice, "mel_spec": test_mel_spec}
    loss = res1d_vocoder.training_step(batch, 0)
    print(loss)
    assert loss.shape == torch.Size([])  # type: ignore
    assert loss.item() > 0  # type: ignore


def test_validation_step(
    res1d_vocoder: Res1DLightningModule, mel_spec_converter: MelSpecConverter
) -> None:
    batch_size = 3
    test_slice = torch.randn((batch_size, 1, 32768))
    test_mel_spec = mel_spec_converter.convert(test_slice)
    batch = {"slice": test_slice, "mel_spec": test_mel_spec}
    res1d_vocoder.validation_step(batch, 0)
