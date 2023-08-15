from torch.utils.data import DataLoader, random_split

from .base import MusicDataModule, MusicDataset
from utils.containers import LearningParameters


class BasicMusicDataModule(MusicDataModule):
    def __init__(
        self, learning_params: LearningParameters, dataset: MusicDataset
    ) -> None:
        super().__init__()
        self.learning_params = learning_params
        self.dataset = dataset

    def setup(self, stage: str) -> None:
        # Split into training and validation
        training_len = int((1 - self.learning_params.val_split) * len(self.dataset))
        val_len = len(self.dataset) - training_len

        self.train_dataset, self.val_dataset = random_split(
            self.dataset, lengths=(training_len, val_len)
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset, batch_size=self.learning_params.batch_size, shuffle=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset, batch_size=self.learning_params.batch_size, shuffle=False
        )
