from typing import Dict
from loaders.base import MelSpecMusicDataset, MusicDataModule

from utils.containers import (
    MelSpecParameters,
    MusicDatasetParameters,
    LearningParameters,
)
from .base import MusicDataset, MusicDataModule, IMusicDatasetFactory
import loaders.data_modules as data_modules
import loaders.datasets as datasets
from models import MelSpecConverterFactory

DATASETS: Dict[str, MusicDataset] = {"music_dataset": datasets.MP3SliceDataset}
MEL_SPEC_DATASETS: Dict[str, MelSpecMusicDataset] = {
    "music_dataset": datasets.MP3MelSpecDataset
}

DATAMODULES: Dict[str, MusicDataModule] = {"basic": data_modules.BasicMusicDataModule}


class MusicDatasetFactory(IMusicDatasetFactory):
    @staticmethod
    def build_music_dataset(dataset_params: MusicDatasetParameters) -> MusicDataset:
        dataset_type = dataset_params.dataset_type
        dataset = DATASETS[dataset_type](dataset_params)
        return dataset

    @staticmethod
    def build_mel_spec_dataset(
        dataset_params: MusicDatasetParameters, mel_spec_params: MelSpecParameters
    ) -> MelSpecMusicDataset:
        base_dataset = MusicDatasetFactory.build_music_dataset(dataset_params)
        mel_spec_converter = MelSpecConverterFactory.build_mel_spec_converter(
            mel_spec_params
        )
        dataset_type = dataset_params.dataset_type
        dataset = MEL_SPEC_DATASETS[dataset_type](base_dataset, mel_spec_converter)
        return dataset

    @staticmethod
    def build_music_data_module(
        dataset_params: MusicDatasetParameters,
        learning_params: LearningParameters,
    ) -> MusicDataModule:
        dataset = MusicDatasetFactory.build_music_dataset(dataset_params)
        datamodule_type = dataset_params.data_module_type
        data_module = DATAMODULES[datamodule_type](learning_params, dataset)
        return data_module

    @staticmethod
    def build_mel_spec_module(
        dataset_params: MusicDatasetParameters,
        learning_params: LearningParameters,
        mel_spec_params: MelSpecParameters,
    ) -> MusicDataModule:
        dataset = MusicDatasetFactory.build_mel_spec_dataset(
            dataset_params, mel_spec_params
        )
        datamodule_type = dataset_params.data_module_type
        data_module = DATAMODULES[datamodule_type](learning_params, dataset)
        return data_module
