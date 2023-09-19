import torch.nn as nn
from torchvision.ops import sigmoid_focal_loss

LOSS_MODULES: dict[str, nn.Module] = {
    "mse": nn.MSELoss(),
    "l1": nn.L1Loss(),
    "huber": nn.SmoothL1Loss(),
    "bce": nn.BCEWithLogitsLoss(),
    "ce": nn.CrossEntropyLoss(),
    "focal": sigmoid_focal_loss,  # type: ignore
}
