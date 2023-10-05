import argparse
from sqlite3 import DataError
from predict import PREDICT_FUNCTIONS

from train import TRAINING_FUNCTIONS
from utils.others import load_config

DEFAULT_CONFIG_PATHS: dict[str, str] = {
    "diffwave": "config/vocoders/config_diffwave.yaml",
    "res1d": "config/vocoders/config_res1d.yaml",
}


def main(args):
    phase: str = args.phase
    resume: str = args.resume
    cfg_path: str = args.config
    model_name: str = args.model
    read_file_path: str = args.read_file
    write_file_path: str = args.write_file

    if cfg_path is None:
        cfg_path = DEFAULT_CONFIG_PATHS[model_name]

    cfg = load_config(cfg_path)
    cfg["learning"]["num_devices"] = args.num_devices

    if phase == "train":
        TRAINING_FUNCTIONS[model_name](cfg, resume)
    elif phase == "predict":
        if read_file_path is None:
            raise DataError("Missing mp3 file path to read")
        if write_file_path is None:
            raise DataError("Missing mp3 file path to write")
        PREDICT_FUNCTIONS[model_name](cfg, read_file_path, write_file_path, resume, 1)
    else:
        raise NotImplementedError(f"Phase {phase} is not implemented yet")


if __name__ == "__main__":
    model_keys = list(TRAINING_FUNCTIONS.keys())
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        default=None,
        help="Configuration yaml file path",
    )
    parser.add_argument(
        "-d", "--num_devices", type=int, default=1, help="Number of devices in use"
    )
    parser.add_argument(
        "-m",
        "--model",
        choices=model_keys,
        help="Current model to train or run inference",
    )
    parser.add_argument(
        "-ph", "--phase", choices=["train", "predict"], help="Train or run the model"
    )
    parser.add_argument(
        "-r", "--resume", type=str, default=None, help="Checkpoint path to resume from"
    )
    parser.add_argument(
        "-rf", "--read_file", type=str, default=None, help="Prediction read file path"
    )
    parser.add_argument(
        "-wf", "--write_file", type=str, default=None, help="Prediction write file path"
    )
    args = parser.parse_args()
    main(args)
