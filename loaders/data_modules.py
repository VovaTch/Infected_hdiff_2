from dataclasses import dataclass
from typing import Protocol

from pytorch_lightning.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS
from torch.utils.data import DataLoader, random_split
import pytorch_lightning as pl

from utils.containers import LearningParameters
from .datasets import MusicDataset


class MusicDataModule(Protocol):
    """
    Base class for a lightning data module implementation, specific for this project.
    """

    def __init__(
        self, learning_params: LearningParameters, dataset: MusicDataset
    ) -> None:
        """
        Constructor for the music data module

        Args:
            learning_params (LearningParameters): Learning parameter object
            dataset (MusicDataset): Dataset object
        """
        ...

    def setup(self, stage: str) -> None:
        """
        The setup method from the lightning module.

        Args:
            *   stage (str): Current stage, supporting 'fit', 'val', 'test', 'pred'.
        """
        ...

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        """
        The train dataloader method from the lightning module.

        Returns:
            *   TRAIN_DATALOADERS: Training dataloader
        """
        ...

    def val_dataloader(self) -> EVAL_DATALOADERS:
        """
        The validation dataloader method from the lightning module.

        Returns:
            EVAL_DATALOADERS: Validation dataloader
        """
        ...


class BasicMusicDataModule(pl.LightningModule):
    """
    Simple data module to be used with standard datasets
    """

    def __init__(
        self, learning_params: LearningParameters, dataset: MusicDataset
    ) -> None:
        """
        Initializer method

        Args:
            learning_params (LearningParameters): Learning parameter object
            dataset (MusicDataset): Dataset object
        """
        super().__init__()
        self.learning_params = learning_params
        self.dataset = dataset

    def setup(self, stage: str) -> None:
        """
        Lightning module setup method

        Args:
            stage (str): Unused in this implementation
        """

        # Split into training and validation
        training_len = int((1 - self.learning_params.val_split) * len(self.dataset))
        val_len = len(self.dataset) - training_len

        self.train_dataset, self.val_dataset = random_split(
            self.dataset, lengths=(training_len, val_len)  # type: ignore
        )

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        """
        Lightning module train_dataloader method

        Returns:
            TRAIN_DATALOADERS: training dataloader object
        """
        return DataLoader(
            self.train_dataset,
            batch_size=self.learning_params.batch_size,
            shuffle=True,
            num_workers=self.learning_params.num_workers,
        )

    def val_dataloader(self) -> EVAL_DATALOADERS:
        """
        Lightning module eval_dataloader method

        Returns:
            EVAL_DATALOADERS: eval dataloader object
        """
        return DataLoader(
            self.val_dataset,
            batch_size=self.learning_params.batch_size,
            shuffle=False,
            num_workers=self.learning_params.num_workers,
        )
