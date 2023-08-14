from abc import abstractmethod
import os
from typing import List, Dict, Any

from torch.utils.data import Dataset

from utils.containers import MusicDatasetParameters


class MusicDataset(Dataset):
    file_list: List[str] = []
    metadata: Dict[str, Any] = {}

    def __init__(self, music_params: MusicDatasetParameters) -> None:
        super().__init__()
        self.music_params = music_params

        for file in os.listdir(self.music_params.data_dir):
            if file.endswith("mp3"):
                self.file_list.append(os.path.join(self.music_params.data_dir, file))

        # Preload
        if self.music_params.preload:
            if os.path.isfile(self.music_params.preload_data_path):
                self._load_data()
