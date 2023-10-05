import torch
import pytest

from models import build_diffwave_diffusion_vocoder, build_mel_spec_converter
from models.diffwave_vocoder import VocoderDiffusionModel
from models.mel_spec_converters import MelSpecConverter
from loss import build_loss_aggregator
from utils.containers import MelSpecParameters
from utils.others import load_config


@pytest.fixture
def vocoder_diffusion_model() -> VocoderDiffusionModel:
    cfg = load_config("config/test_config.yaml")
    loss_aggregator = build_loss_aggregator(cfg)
    return build_diffwave_diffusion_vocoder(cfg, loss_aggregator=loss_aggregator)


@pytest.fixture
def mel_spec_converter() -> MelSpecConverter:
    cfg = load_config("config/test_config.yaml")
    mel_spec_params = MelSpecParameters(**cfg["image_mel_spec_params"])
    return build_mel_spec_converter("simple", mel_spec_params)


def test_forward(
    vocoder_diffusion_model: VocoderDiffusionModel, mel_spec_converter: MelSpecConverter
) -> None:
    batch_size = 3
    test_noisy_slice = torch.randn((batch_size, 32768))
    test_mel_spec = mel_spec_converter.convert(test_noisy_slice)
    test_time_steps = torch.randint(0, 10, [3])
    test_dict = {
        "noisy_slice": test_noisy_slice,
        "mel_spec": test_mel_spec,
        "time_step": test_time_steps,
    }
    outputs = vocoder_diffusion_model.forward(test_dict)
    assert outputs["noise_pred"].shape == test_dict["noisy_slice"].unsqueeze(1).shape


def test_denoising(
    vocoder_diffusion_model: VocoderDiffusionModel, mel_spec_converter: MelSpecConverter
) -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    vocoder_diffusion_model = vocoder_diffusion_model.to(device)
    mel_spec_converter.mel_spec = mel_spec_converter.mel_spec.to(device)
    batch_size = 1
    test_noisy_slice = torch.randn((batch_size, 32768)).to(device)
    test_mel_spec = mel_spec_converter.convert(test_noisy_slice)
    input_slice = {"noisy_slice": test_noisy_slice}
    mel_spec_dict = {"mel_spec": test_mel_spec}
    outputs = vocoder_diffusion_model.denoise(input_slice, mel_spec_dict)
    assert outputs["denoised_slice"].shape == test_noisy_slice.shape
