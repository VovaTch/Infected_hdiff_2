import os
import json

from torchaudio.transforms import MelSpectrogram

from .base import MelSpecMusicDataset
from utils.containers import MelSpecParameters, MusicDatasetParameters


class MP3MelSpecDataset(MelSpecMusicDataset):
    def __init__(
        self, dataset_params: MusicDatasetParameters, mel_spec_params: MelSpecParameters
    ) -> None:
        super().__init__(dataset_params, mel_spec_params)

        self.preload = dataset_params.preload

        # Check if the data is loaded
        if os.path.exists(dataset_params.data_dir):
            self.buffer = self._load_data(dataset_params.data_dir)
        else:
            os.makedirs(dataset_params.data_dir)
            self.buffer = self._generate_data()
            self._dump_data(dataset_params.data_dir)

    def _generate_data(self):
        image_data_path = os.path.join(self.dataset_params.data_dir, "melspec_img")
        os.makedirs(image_data_path)
        metadata_json_path = os.path.join(self.dataset_params.data_dir, "metadata.json")

        # Create mp3 file list
        self.file_list = []
        for file in os.listdir(self.dataset_params.data_dir):
            if file.endswith("mp3"):
                self.file_list.append(os.path.join(self.dataset_params.data_dir, file))

        # Initialize mel spectrogram
        mel_spec = MelSpectrogram(self.mel_spec_params.__dict__)
