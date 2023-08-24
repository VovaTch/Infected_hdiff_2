import unittest

from torch.utils.data import DataLoader

from models import MelSpecConverterFactory
from utils.others import load_config
from utils.containers import MelSpecParameters, MusicDatasetParameters
from loaders import MusicDatasetFactory


class TestMelSpecConvertor(unittest.TestCase):
    def setUp(self) -> None:
        self.cfg = load_config("config/test_config.yaml")
        dataset_params = MusicDatasetParameters(**self.cfg["dataset"])
        self.slice_dataset = MusicDatasetFactory.build_music_dataset(dataset_params)
        self.slice_dataloader = DataLoader(
            self.slice_dataset, batch_size=1, shuffle=True
        )

    def test_simple_mel_spec(self):
        for batch_ind in self.slice_dataloader:
            batch = batch_ind
            break

        mel_spec_converter = MelSpecConverterFactory.build_mel_spec_converter(
            type="simple",
            mel_spec_params=MelSpecParameters(**self.cfg["image_mel_spec_params"]),
        )
        mel_spec = mel_spec_converter.convert(batch["slice"])
        self.assertEqual(mel_spec.shape[0], 1)

    def test_scaled_image_mel_spec(self):
        for batch_ind in self.slice_dataloader:
            batch = batch_ind
            break

        mel_spec_converter = MelSpecConverterFactory.build_mel_spec_converter(
            type="scaled_image",
            mel_spec_params=MelSpecParameters(**self.cfg["image_mel_spec_params"]),
        )
        mel_spec = mel_spec_converter.convert(batch["slice"])
        self.assertEqual(mel_spec.shape[0], 1)


if __name__ == "__main__":
    unittest.main()
