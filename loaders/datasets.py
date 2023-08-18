import os
from typing import Dict, Any
import json

import torch
import torch.nn.functional as F
import torchaudio
import tqdm

from .base import MusicDataset
from utils.containers import MusicDatasetParameters


class MP3MelSpecDataset(MusicDataset):
    def __init__(self, dataset_params: MusicDatasetParameters) -> None:
        super().__init__(dataset_params)

        self.preload = dataset_params.preload
        self.slice_length = dataset_params.slice_length
        self.sample_rate = dataset_params.sample_rate
        self.device = dataset_params.device

        # Check if the data is loaded
        slices_path = os.path.join(dataset_params.data_dir, "slices")
        if os.path.exists(slices_path):
            self.buffer = self._load_data(dataset_params.data_dir)
        else:
            os.makedirs(dataset_params.data_dir)
            self._generate_data()
            self._dump_data(dataset_params.data_dir)

    def _generate_data(self):
        # Create mp3 file list
        self.file_list = []
        for file in os.listdir(self.dataset_params.data_dir):
            if file.endswith("mp3"):
                self.file_list.append(os.path.join(self.dataset_params.data_dir, file))

        # Initialize mel spectrogram
        for file in tqdm.tqdm(self.file_list, "Loading data from files..."):
            file_data = self._load_data_from_track(file)

            # Append to the buffer
            for key, value in file_data.items():
                if key in self.buffer:
                    self.buffer[key] += value
                else:
                    self.buffer[key] = value

    def _load_data_from_track(self, file: str) -> Dict[str, Any]:
        long_data, sr = torchaudio.load(file, format="mp3")
        long_data = self._resample_if_necessary(long_data, sr)
        long_data = self._mix_down_if_necessary(long_data)
        long_data = self._right_pad_if_necessary(long_data)
        slices = long_data.view((-1, 1, self.slice_length))
        slice_file_name = "slices_" + file[:-4] + ".pt"

        # Make the data fit into the buffer
        slices_list = [ind_slice.to(self.device) for ind_slice in slices]
        track_name = [file for _ in range(slices.shape[0])]
        slice_file_name_list = [slice_file_name for _ in range(slices.shape[0])]
        slice_idx = [idx for idx in range(slices.shape[0])]
        slice_init_time = [idx * self.slice_length for idx in range(slices.shape[0])]

        # Aggregate all data in a dictionary
        data = {
            "slice": slices_list,
            "slice_file_name": slice_file_name_list,
            "track_name": track_name,
            "slice_idx": slice_idx,
            "slice_init_time": slice_init_time,
        }

        return data

    def _resample_if_necessary(self, signal: torch.Tensor, sr: int):
        if sr != self.sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
            signal = resampler(signal)
        return signal

    def _mix_down_if_necessary(self, signal: torch.Tensor):
        if signal.shape[0] > 1:
            signal = torch.mean(signal, dim=0, keepdim=True)
        return signal

    def _right_pad_if_necessary(self, signal: torch.Tensor):
        length_signal = signal.shape[1]
        if length_signal % self.slice_length != 0:
            num_missing_samples = self.slice_length - length_signal % self.slice_length
            last_dim_padding = (0, num_missing_samples)
            signal = F.pad(signal, last_dim_padding)
        return signal

    def _dump_data(self, path: str) -> None:
        # Make slice folder
        slice_data_path = os.path.join(path, "slices")
        os.makedirs(slice_data_path)

        # Make metadata folder
        metadata_path = os.path.join(path, "metadata.json")
        metadata_for_json = [
            {key: value[idx] for (key, value) in self.buffer.items() if key != "slice"}
            for idx in range(len(self.buffer["slices"]))
        ]
        with open(metadata_path, "w") as f:
            f.write(metadata_for_json)

        for file in tqdm.tqdm(self.file_list, "Saving slices as .pt files..."):
            aggregate_slices = [
                self.buffer["slices"][idx]
                for idx in range(len(self.buffer["slices"]))
                if self.buffer["track_name"] == file
            ]
            aggregate_slices_torch = torch.stack(aggregate_slices, dim=0)
            slice_file_name = "slices_" + file[:-4] + ".pt"

            # Saving this as a .pt file
            torch.save(aggregate_slices_torch, slice_file_name)

    def _load_data(self, path: str) -> None:
        json_file_path = os.path.join(path, "metadata.json")
        with open(json_file_path, "r") as f:
            metadata = json.load(f)

        # Reorganize the metadata file as a buffer
        for datapoint in metadata:
            for key, value in datapoint:
                # Convert to the correct device
                if isinstance(value, torch.Tensor):
                    value = value.to(self.device)

                # Input in the buffer
                if key not in self.buffer:
                    self.buffer[key] = [value]
                else:
                    self.buffer[key] += [value]

    def __getitem__(self, index: int) -> Dict[str, Any]:
        datapoint = {key: value[index] for (key, value) in self.buffer.items()}
        for value in datapoint.values():
            if isinstance(value, torch.Tensor):
                value = value.to(self.device)

        return datapoint
