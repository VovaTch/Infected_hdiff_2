import argparse

from loaders.build import build_music_data_module
from utils.containers import LearningParameters, MusicDatasetParameters
from utils.others import load_config


def main(args):
    cfg_path = args.config
    cfg = load_config(cfg_path)

    learning_params = LearningParameters(**cfg["learning"])
    learning_params.batch_size = 3
    dataset_params = MusicDatasetParameters(**cfg["dataset"])

    data_module = build_music_data_module(dataset_params, learning_params)
    data_module.setup("fit")
    train_dataloader = data_module.train_dataloader()
    for batch in train_dataloader:
        print(batch)
        break


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Script to create a dataset for low level music reconstruction training."
    )
    parser.add_argument(
        "-c", "--config", type=str, default="config/vocoders/config_diffwave.yaml"
    )
    args = parser.parse_args()
    main(args)
