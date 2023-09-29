from utils.containers import (
    MelSpecParameters,
    MusicDatasetParameters,
    LearningParameters,
)
import loaders.data_modules as data_modules
import loaders.datasets as datasets
from models.build import build_mel_spec_converter
from .datasets import MelSpecDataset, MusicDataset
from .data_modules import MusicDataModule

DATASETS: dict[str, type[MusicDataset]] = {
    "music_dataset": datasets.MP3SliceDataset,
    "test": datasets.TestDataset,
}
MEL_SPEC_DATASETS: dict[str, type[MelSpecDataset]] = {
    "music_dataset": datasets.MP3MelSpecDataset,
    "test": datasets.TestMelSpecDataset,  # type: ignore
}

DATAMODULES: dict[str, type[MusicDataModule]] = {
    "basic": data_modules.BasicMusicDataModule
}


def build_music_dataset(dataset_params: MusicDatasetParameters) -> MusicDataset:
    """
    Music dataset building factory function

    Args:
        dataset_params (MusicDatasetParameters): Dataset parameters object

    Returns:
        MusicDataset: Music dataset object
    """
    dataset_type = dataset_params.dataset_type
    dataset = DATASETS[dataset_type](dataset_params)
    return dataset


def build_mel_spec_dataset(
    dataset_params: MusicDatasetParameters, mel_spec_params: MelSpecParameters
) -> MelSpecDataset:
    """
    Music dataset with mel-spec data building factory function

    Args:
        dataset_params (MusicDatasetParameters): Dataset parameters object
        mel_spec_params (MelSpecParameters): Mel Spectrogram parameters object

    Returns:
        MelSpecDataset: Music dataset with mel-spec data
    """
    base_dataset = build_music_dataset(dataset_params)
    mel_spec_converter = build_mel_spec_converter("simple", mel_spec_params)
    dataset_type = dataset_params.dataset_type
    dataset = MEL_SPEC_DATASETS[dataset_type](base_dataset, mel_spec_converter)
    return dataset


def build_music_data_module(
    dataset_params: MusicDatasetParameters,
    learning_params: LearningParameters,
) -> MusicDataModule:
    """
    Music lightning data module building function

    Args:
        dataset_params (MusicDatasetParameters): Dataset parameters object
        learning_params (LearningParameters): Learning parameters object

    Returns:
        MusicDataModule: Music lightning data module object
    """
    dataset = build_music_dataset(dataset_params)
    datamodule_type = dataset_params.data_module_type
    data_module = DATAMODULES[datamodule_type](learning_params, dataset)
    return data_module


def build_mel_spec_module(
    dataset_params: MusicDatasetParameters,
    learning_params: LearningParameters,
    mel_spec_params: MelSpecParameters,
) -> MusicDataModule:
    """
    Music + mel spectrogram lightning data module building function

    Args:
        dataset_params (MusicDatasetParameters): Dataset parameters object
        learning_params (LearningParameters): Learning parameters object
        mel_spec_params (MelSpecParameters): Mel Spectrogram parameter object

    Returns:
        MusicDataModule: Music lightning data module object
    """
    dataset = build_mel_spec_dataset(dataset_params, mel_spec_params)
    datamodule_type = dataset_params.data_module_type
    data_module = DATAMODULES[datamodule_type](learning_params, dataset)  # type: ignore
    return data_module
