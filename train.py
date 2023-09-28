from typing import Any, Callable

from loaders.build import build_mel_spec_module
from loss.factory import build_loss_aggregator
from models.build import build_diffwave_diffusion_vocoder
from utils.containers import parse_cfg_for_vocoder
from utils.trainer import initialize_trainer


def train_diffwave(cfg: dict[str, Any], weights_path: str | None = None) -> None:
    learning_params, dataset_params, mel_spec_params = parse_cfg_for_vocoder(cfg)
    loss_aggregator = build_loss_aggregator(cfg)
    model = build_diffwave_diffusion_vocoder(
        cfg, loss_aggregator=loss_aggregator, weights_path=weights_path
    )
    data_module = build_mel_spec_module(
        dataset_params, learning_params, mel_spec_params
    )
    trainer = initialize_trainer(learning_params)

    # Train
    trainer.fit(model, datamodule=data_module)  # type: ignore


TRAINING_FUNCTIONS: dict[str, Callable[[dict[str, Any], str], None]] = {
    "diffwave": train_diffwave
}
