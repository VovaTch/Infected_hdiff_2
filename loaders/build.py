from typing import Dict

from loaders.base import MusicDataModule, MusicDataset
from utils.containers import MusicDatasetParameters, LearningParameters
from .base import MusicDataset, MusicDataModule, IMusicDatasetFactory
import loaders.data_modules as data_modules
import loaders.datasets as datasets

DATASETS: Dict[str, MusicDataset] = {"music_dataset": datasets.MP3MelSpecDataset}

DATAMODULES: Dict[str, MusicDataModule] = {"basic": data_modules.BasicMusicDataModule}


class MusicDatasetFactory(IMusicDatasetFactory):
    @staticmethod
    def build_music_dataset(dataset_params: MusicDatasetParameters) -> MusicDataset:
        dataset_type = dataset_params.dataset_type
        dataset = DATASETS[dataset_type](dataset_params)
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
