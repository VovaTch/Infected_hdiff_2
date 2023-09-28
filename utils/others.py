from typing import Any

import yaml


def load_config(path: str) -> dict[str, Any]:
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    return cfg
