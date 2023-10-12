from typing import Any, Callable, Optional

import torch
from torch.utils.data import TensorDataset, DataLoader
import torchaudio
import tqdm
from models.base import BaseLightningModule

from models.build import (
    build_diffwave_diffusion_vocoder,
    build_mel_spec_converter,
    build_res1d_vocoder,
)
from utils.containers import MelSpecParameters
from utils.slices import create_slices_from_file


@torch.no_grad()
def predict_diffwave(
    cfg: dict[str, Any],
    read_mp3_file_path: str,
    write_mp3_file_path: str,
    weights_path: str | None = None,
    batch_size: int = 1,
) -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    required_sampling_rate = cfg["dataset"]["sample_rate"]
    slice_length = cfg["dataset"]["slice_length"]
    mel_spec_params = MelSpecParameters(**cfg["image_mel_spec_params"])
    model = build_diffwave_diffusion_vocoder(cfg, weights_path=weights_path).to(device)
    full_file_slices = create_slices_from_file(
        read_mp3_file_path, required_sampling_rate, slice_length
    )

    # Create the mel-spec converter
    mel_spec_converter = build_mel_spec_converter("simple", mel_spec_params)
    mel_spec_converter.mel_spec = mel_spec_converter.mel_spec.to(device)

    # Create the tensor dataset and load from it for inference
    slice_dataset = TensorDataset(full_file_slices)
    slice_dataloader = DataLoader(slice_dataset, batch_size=batch_size)
    reconstructed_slice = torch.zeros(
        (0, full_file_slices.shape[1], full_file_slices.shape[2])
    ).to(device)
    for batch in tqdm.tqdm(
        slice_dataloader, desc=f"Processing the music file {read_mp3_file_path}..."
    ):
        slice: torch.Tensor = batch[0].squeeze(1)
        mel_spec = mel_spec_converter.convert(slice).squeeze(1)
        noisy_input = {"noisy_slice": torch.randn_like(slice).float()}
        cond = {"mel_spec": mel_spec}
        reconstructed_slice_ind = model.denoise(noisy_input, cond)["denoised_slice"]
        reconstructed_slice = torch.cat(
            (reconstructed_slice, reconstructed_slice_ind.unsqueeze(1)), dim=0
        )

    long_reconstructed_slice = reconstructed_slice.view((1, -1))
    torchaudio.save(write_mp3_file_path, long_reconstructed_slice.cpu().detach(), required_sampling_rate, format="mp3")  # type: ignore


@torch.no_grad()
def predict_res1d(
    cfg: dict[str, Any],
    read_mp3_file_path: str,
    write_mp3_file_path: str,
    weights_path: str | None = None,
    batch_size: int = 1,
) -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    required_sampling_rate = cfg["dataset"]["sample_rate"]
    slice_length = cfg["dataset"]["slice_length"]
    mel_spec_params = MelSpecParameters(**cfg["image_mel_spec_params"])
    model = build_res1d_vocoder(cfg, weights_path=weights_path).to(device)
    full_file_slices = create_slices_from_file(
        read_mp3_file_path, required_sampling_rate, slice_length
    )

    # Create the mel-spec converter
    mel_spec_converter = build_mel_spec_converter("simple", mel_spec_params)
    mel_spec_converter.mel_spec = mel_spec_converter.mel_spec.to(device)

    # Create the tensor dataset and load from it for inference
    slice_dataset = TensorDataset(full_file_slices)
    slice_dataloader = DataLoader(slice_dataset, batch_size=batch_size)
    reconstructed_slice = torch.zeros(
        (0, full_file_slices.shape[1], full_file_slices.shape[2])
    ).to(device)
    for batch in tqdm.tqdm(
        slice_dataloader, desc=f"Processing the music file {read_mp3_file_path}..."
    ):
        slice: torch.Tensor = batch[0].squeeze(1)
        mel_spec = mel_spec_converter.convert(slice).squeeze(1)
        inputs = {"mel_spec": mel_spec.to(device)}
        reconstructed_slice_ind = model.forward(inputs)["slice"]
        reconstructed_slice = torch.cat(
            (reconstructed_slice, reconstructed_slice_ind), dim=0
        )

    long_reconstructed_slice = reconstructed_slice.view((1, -1))
    torchaudio.save(write_mp3_file_path, long_reconstructed_slice.cpu().detach(), required_sampling_rate, format="mp3")  # type: ignore


PREDICT_FUNCTIONS: dict[str, Callable[[dict[str, Any], str, str, str, int], None]] = {
    "diffwave": predict_diffwave,
    "res1d": predict_res1d,
}
