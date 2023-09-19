from typing import Callable
import torch

TRANSFORM_FUNCS: dict[str, Callable[[torch.Tensor], torch.Tensor]] = {
    "none": lambda x: x,
    "tanh": lambda x: torch.tanh(x),
    "sigmoid": lambda x: torch.sigmoid(x),
}
