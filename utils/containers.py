from dataclasses import dataclass

import torch


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
    num_devices: int = 1
    num_workers: int = 0

    # Scheduler settings
    loss_monitor: str = "step"
    interval: str = "training_total_loss"
    frequency: int = 1


@dataclass
class MusicDatasetParameters:
    sample_rate: int
    data_dir: str
    slice_length: int
    preload: bool
    device: str = "cpu"
    preload_data_path: str
    preload_metadata_path: str
