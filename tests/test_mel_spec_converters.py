import unittest
import pytest

from torch.utils.data import DataLoader

from models import build_mel_spec_converter
from utils.others import load_config
from utils.containers import MelSpecParameters, MusicDatasetParameters
from loaders import build_music_dataset


@pytest.fixture
def get_slice_dataset():
    cfg = load_config("config/test_config.yaml")
    dataset_params = MusicDatasetParameters(**cfg["dataset"])
    slice_dataset = build_music_dataset(dataset_params)
    slice_dataloader = DataLoader(slice_dataset, batch_size=1, shuffle=True)  # type: ignore
    return cfg, slice_dataloader


def test_simple_mel_spec(get_slice_dataset):
    cfg, slice_dataloader = get_slice_dataset

    for batch_ind in slice_dataloader:
        batch = batch_ind
        break

    mel_spec_converter = build_mel_spec_converter(
        type="simple",
        mel_spec_params=MelSpecParameters(**cfg["image_mel_spec_params"]),
    )
    mel_spec = mel_spec_converter.convert(batch["slice"])  # type: ignore
    assert mel_spec.shape[0] == 1


def test_scaled_mel_spec(get_slice_dataset):
    cfg, slice_dataloader = get_slice_dataset

    for batch_ind in slice_dataloader:
        batch = batch_ind
        break

    mel_spec_converter = build_mel_spec_converter(
        type="scaled_image",
        mel_spec_params=MelSpecParameters(**cfg["image_mel_spec_params"]),
    )
    mel_spec = mel_spec_converter.convert(batch["slice"])  # type: ignore
    assert mel_spec.shape[0] == 1
