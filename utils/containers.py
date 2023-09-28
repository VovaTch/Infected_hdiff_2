from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch

if TYPE_CHECKING:
    from models.base import DiffusionScheduler


@dataclass
class VocoderOutput:
    waveform: torch.Tensor


@dataclass
class TransformerDecoderOutput:
    indices: torch.Tensor
    tokens: torch.Tensor
    logits: torch.Tensor


@dataclass
class LearningParameters:
    # Name
    model_name: str

    # Learning settings
    learning_rate: float
    weight_decay: float
    batch_size: int
    epochs: int
    beta_ema: float
    gradient_clip: float
    save_path: str
    eval_split_factor: float
    amp: bool
    val_split: float
    num_devices: int = 1
    num_workers: int = 0

    # Scheduler settings
    loss_monitor: str = "step"
    interval: str = "training_total_loss"
    frequency: int = 1


@dataclass
class MusicDatasetParameters:
    dataset_type: str
    data_module_type: str
    sample_rate: int
    data_dir: str
    slice_length: int
    preload: bool
    preload_data_dir: str
    device: str = "cpu"


@dataclass
class DiffusionParameters:
    scheduler: "DiffusionScheduler"
    num_steps: int


@dataclass
class MelSpecParameters:
    n_fft: int
    hop_length: int
    n_mels: int
    power: float
    f_min: float
    pad: int
    pad_mode: str = "reflect"
    norm: str = "slaney"
    mel_scale: str = "htk"


def parse_cfg_for_vocoder(
    cfg: dict[str, Any]
) -> tuple[LearningParameters, MusicDatasetParameters, MelSpecParameters]:
    learning_params = LearningParameters(**cfg["learning"])
    dataset_params = MusicDatasetParameters(**cfg["dataset"])
    mel_spec_params = MelSpecParameters(**cfg["image_mel_spec_params"])
    return learning_params, dataset_params, mel_spec_params
