import unittest

import yaml

from loaders import MusicDatasetFactory
from utils.containers import MusicDatasetParameters, LearningParameters


class TestLoaders(unittest.TestCase):
    def setUp(self) -> None:
        # Load config
        cfg_path = "config/config.yaml"
        with open(cfg_path, "r") as f:
            self.cfg = yaml.safe_load(f)

        learning_params = LearningParameters(**self.cfg["learning"])
        learning_params.batch_size = 3
        dataset_params = MusicDatasetParameters(**self.cfg["dataset"])

        self.data_module = MusicDatasetFactory.build_music_data_module(
            dataset_params, learning_params
        )

    def test_data_loading(self):
        self.data_module.setup("fit")
        train_dataloader = self.data_module.train_dataloader()
        for batch in train_dataloader:
            break
        self.assertEqual(len(batch["slice"]), 3)


if __name__ == "__main__":
    unittest.main()
