from abc import ABC, abstractmethod
from typing import Dict, Any
from pytorch_lightning.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS

from torch.utils.data import Dataset
import pytorch_lightning as pl

from utils.containers import (
    MusicDatasetParameters,
    LearningParameters,
    MelSpecParameters,
)


class MusicDataset(Dataset):
    """
    Music dataset interface, inherits from Pytorch's Dataset class.
    """

    buffer: Dict[str, Any] = {}

    def __init__(self, dataset_params: MusicDatasetParameters) -> None:
        """
        Args:
            dataset_params (MusicDatasetParameters): Dataset parameters object
        """
        super().__init__()
        self.dataset_params = dataset_params

    @abstractmethod
    def _load_data(self, path: str) -> None:
        """
        Loads data from external dataset files into an internal buffer.

        Args:
            path (str): Dataset path
        """
        ...

    @abstractmethod
    def _save_data(self, data: Dict[str, Any]) -> None:
        """
        Saves data from an external dictionary into the internal representation

        Args:
            data (Dict[str, Any]): Data dictionary
        """
        ...

    @abstractmethod
    def _dump_data(self, path: str) -> None:
        """
        Saves data from internal dictionary into a folder path

        Args:
            path (str): Folder path to save data
        """
        ...

    def __len__(self) -> int:
        """
        Returns:
            int: Dataset length
        """

        if self.buffer is None or len(self.buffer) == 0:
            return 0
        else:
            return len(list(self.buffer.values())[0])

    @abstractmethod
    def __getitem__(self, index: int) -> Dict[str, Any]:
        """
        Gets an item, must be implemented to extend the Dataset class

        Args:
            index (int): Dataset index

        Returns:
            Dict[str, Any]: Output dictionary
        """
        ...


class MelSpecMusicDataset(MusicDataset):
    """
    Extension of the MusicDataset abstract class, designed to get also the mel-spectrogram parameters object.
    """

    def __init__(
        self, dataset_params: MusicDatasetParameters, mel_spec_params: MelSpecParameters
    ) -> None:
        """
        Args:
            *   dataset_params (MusicDatasetParameters): Dataset parameters object
            *   mel_spec_params (MelSpecParameters): mel-spectrogram parameters object
        """
        super().__init__(dataset_params)
        self.mel_spec_params = mel_spec_params


class MusicDataModule(pl.LightningDataModule):
    """
    Base class for a lightning data module implementation, specific for this project.
    """

    def __init__(
        self, learning_params: LearningParameters, dataset: MusicDataset
    ) -> None:
        """
        Args:
            *   learning_params (LearningParameters): Learning parameters object
            *   dataset (MusicDataset): Dataset parameters object
        """
        super().__init__()
        self.learning_params = learning_params
        self.dataset = dataset

    @abstractmethod
    def setup(self, stage: str) -> None:
        """
        The setup method from the lightning module.

        Args:
            *   stage (str): Current stage, supporting 'fit', 'val', 'test', 'pred'.
        """
        ...

    @abstractmethod
    def train_dataloader(self) -> TRAIN_DATALOADERS:
        """
        The train dataloader method from the lightning module.

        Returns:
            *   TRAIN_DATALOADERS: Training dataloader
        """
        ...

    @abstractmethod
    def val_dataloader(self) -> EVAL_DATALOADERS:
        """
        The validation dataloader method from the lightning module.

        Returns:
            EVAL_DATALOADERS: Validation dataloader
        """
        ...


class IMusicDatasetFactory(ABC):
    @staticmethod
    @abstractmethod
    def build_music_dataset(dataset_params: MusicDatasetParameters) -> MusicDataset:
        ...

    @staticmethod
    @abstractmethod
    def build_music_data_module(
        dataset_params: MusicDatasetParameters, learning_params: LearningParameters
    ) -> MusicDataModule:
        ...