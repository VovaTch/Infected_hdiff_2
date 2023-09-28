import argparse

from train import TRAINING_FUNCTIONS
from utils.others import load_config


def main(args):
    phase: str = args.phase
    resume: str = args.resume
    cfg_path: str = args.config
    model_name: str = args.model

    cfg = load_config(cfg_path)
    cfg["learning"]["num_devices"] = args.num_devices

    if phase == "train":
        TRAINING_FUNCTIONS[model_name](cfg, resume)
    else:
        raise NotImplementedError()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        default="config/config.yaml",
        help="Configuration yaml file path",
    )
    parser.add_argument(
        "-d", "--num_devices", type=int, default=1, help="Number of devices in use"
    )
    parser.add_argument(
        "-m",
        "--model",
        choices=["diffwave"],
        help="Current model to train or run inference",
    )
    parser.add_argument(
        "-ph", "--phase", choices=["train", "run"], help="Train or run the model"
    )
    parser.add_argument(
        "-r", "--resume", type=str, default=None, help="Checkpoint path to resume from"
    )
    args = parser.parse_args()
    main(args)
