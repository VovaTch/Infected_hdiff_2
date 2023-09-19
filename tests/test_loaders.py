from typing import Any

import yaml
import pytest

from loaders import (
    build_mel_spec_module,
    build_music_data_module,
)
from utils.containers import (
    MelSpecParameters,
    MusicDatasetParameters,
    LearningParameters,
)


@pytest.fixture
def get_cfg() -> dict[str, Any]:
    cfg_path = "config/test_config.yaml"
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)
    return cfg


def test_music_data_module(get_cfg):
    cfg = get_cfg

    learning_params = LearningParameters(**cfg["learning"])
    learning_params.batch_size = 3
    dataset_params = MusicDatasetParameters(**cfg["dataset"])

    data_module = build_music_data_module(dataset_params, learning_params)
    data_module.setup("fit")
    train_dataloader = data_module.train_dataloader()
    for batch in train_dataloader:
        break

    assert len(batch["slice"]) == 3  # type: ignore


def test_mel_spec_data_module(get_cfg):
    cfg = get_cfg

    learning_params = LearningParameters(**cfg["learning"])
    learning_params.batch_size = 3
    dataset_params = MusicDatasetParameters(**cfg["dataset"])
    mel_spec_params = MelSpecParameters(**cfg["image_mel_spec_params"])

    data_module = build_mel_spec_module(
        dataset_params, learning_params, mel_spec_params
    )
    data_module.setup("fit")
    train_dataloader = data_module.train_dataloader()
    for batch in train_dataloader:
        break

    assert len(batch["slice"]) == 3  # type: ignore
