from typing import Any, TYPE_CHECKING, Optional

from diffwave.model import DiffWave
from diffwave.params import AttrDict
import numpy as np
from pytorch_lightning.utilities.types import STEP_OUTPUT
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns

from utils.containers import (
    DiffusionParameters,
    LearningParameters,
    MelSpecParameters,
    MusicDatasetParameters,
)
from .base import BaseDiffusionModel

if TYPE_CHECKING:
    from loss.aggregators import LossAggregator, LossOutput


class DiffwaveWrapper(nn.Module):
    def __init__(
        self,
        learning_params: LearningParameters,
        mel_spec_params: MelSpecParameters,
        dataset_params: MusicDatasetParameters,
    ) -> None:
        super().__init__()
        self.params = AttrDict(  # Training params
            batch_size=learning_params.batch_size,
            learning_rate=learning_params.learning_rate,
            max_grad_norm=None,
            # Data params
            sample_rate=dataset_params.sample_rate,
            n_mels=mel_spec_params.n_mels,
            n_fft=mel_spec_params.n_fft,
            hop_samples=mel_spec_params.hop_length,
            crop_mel_frames=62,  # Probably an error in paper.
            # Model params
            residual_layers=30,
            residual_channels=64,
            dilation_cycle_length=10,
            unconditional=False,
            noise_schedule=np.linspace(1e-4, 0.05, 50).tolist(),
            inference_noise_schedule=[0.0001, 0.001, 0.01, 0.05, 0.2, 0.5],
            # unconditional sample len
            audio_len=dataset_params.slice_length,  # unconditional_synthesis_samples)
        )
        self.model = DiffWave(AttrDict(self.params))

    def forward(self, noisy_input: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        outputs = self.model(
            noisy_input["noisy_slice"],
            noisy_input["mel_spec"],
            noisy_input["time_step"],
        )
        return {"noise_pred": outputs}


class VocoderDiffusionModel(BaseDiffusionModel):
    def __init__(
        self,
        base_model: nn.Module,
        learning_params: LearningParameters,
        diffusion_params: DiffusionParameters,
        loss_aggregator: Optional["LossAggregator"] = None,
        optimizer: torch.optim.Optimizer | None = None,
        scheduler: Any = None,
        **kwargs,
    ) -> None:
        super().__init__(
            base_model,
            learning_params,
            diffusion_params,
            loss_aggregator,
            optimizer,
            scheduler,
            **kwargs,
        )
        if not optimizer:
            self.optimizer = torch.optim.AdamW(
                self.parameters(),
                lr=learning_params.learning_rate,
                weight_decay=learning_params.weight_decay,
            )

    def forward(self, x: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        outputs: dict[str, torch.Tensor] = self.model(x)
        return outputs

    def set_optimizer(self, optimizer: torch.optim.Optimizer) -> None:
        self.optimizer = optimizer

    def training_step(
        self, batch: dict[str, torch.Tensor], batch_idx: int
    ) -> STEP_OUTPUT:
        # Ensure there is loss and an optimizer during training
        if not self.loss_aggregator:
            raise ValueError("Must include a loss aggregator during training")
        if not self.optimizer:
            raise ValueError("Must include an optimizer during training")

        updated_inputs, slice_outputs = self._step(batch)
        loss = self.loss_aggregator.compute(slice_outputs, updated_inputs)

        self._log_losses("training", loss)
        return loss.total

    def validation_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> None:
        updated_inputs, slice_outputs = self._step(batch)

        if not self.loss_aggregator:
            return
        loss = self.loss_aggregator.compute(slice_outputs, updated_inputs)

        for loss_key, loss_value in loss.individuals.items():
            self.log(loss_key, loss_value)

        self._log_losses("validation", loss)

    def _step(
        self, batch: dict[str, torch.Tensor]
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        # Create noised input
        sampled_time_steps = torch.randint(
            0,
            self.diffusion_params.num_steps,
            [batch["slice"].shape[0]],
            device=self.device,
        ).to(batch["slice"].device)
        batch["mel_spec"] = batch["mel_spec"].squeeze(1)
        batch["slice"] = batch["slice"].squeeze(1)
        noisy_input = self.sample_timestep(batch, sampled_time_steps)
        updated_inputs = {**noisy_input, **batch, "time_step": sampled_time_steps}

        slice_outputs = self.forward(updated_inputs)
        slice_outputs["noise_pred"] = slice_outputs["noise_pred"].squeeze(1)
        return updated_inputs, slice_outputs

    def _log_losses(self, phase_name: str, loss: "LossOutput") -> None:
        for loss_key, loss_value in loss.individuals.items():
            self.log(f"{phase_name}_{loss_key}", loss_value)

        self.log(f"{phase_name}_total_loss", loss.total, prog_bar=True)

    def sample_timestep(
        self,
        x: dict[str, torch.Tensor],
        t: torch.Tensor,
        cond: dict[str, torch.Tensor] = {},
        verbose: bool = False,
    ) -> dict[str, torch.Tensor]:
        noise_scale = self.diffusion_params.scheduler.alphas_cumprod[t].unsqueeze(1)
        noise_scale_sqrt = noise_scale**0.5
        noise = torch.randn_like(x["slice"])
        noisy_slice = noise_scale_sqrt * x["slice"] + (1.0 - noise_scale) ** 0.5 * noise
        return {"noisy_slice": noisy_slice, "noise": noise, "noise_scale": noise_scale}

    @torch.no_grad()
    def denoise(
        self,
        noisy_input: dict[str, torch.Tensor],
        cond: dict[str, torch.Tensor] = {},
        show_process_plots: bool = False,
    ) -> dict[str, torch.Tensor]:
        with torch.no_grad():
            betas = self.diffusion_params.scheduler.betas
            alphas = self.diffusion_params.scheduler.alphas
            alphas_cumprod = self.diffusion_params.scheduler.alphas_cumprod

            # This is from the code in the repo
            time_series = []
            for s in range(betas.shape[0]):
                for t in range(betas.shape[0]):
                    if alphas_cumprod[t + 1] <= alphas_cumprod[s] <= alphas_cumprod[t]:
                        widdle = (
                            alphas_cumprod[t] ** 0.5 - alphas_cumprod[s] ** 0.5
                        ) / (alphas_cumprod[t] ** 0.5 - alphas_cumprod[t + 1] ** 0.5)
                        time_series.append(t + widdle)
                        break
            time_series = torch.tensor(time_series).to(self.device)

            conditioned_mel_spec = cond["mel_spec"].clone().to(self.device)
            denoised_slice = noisy_input["noisy_slice"].clone().to(self.device)

            # TODO: I hate the naming here, just copying from the repo
            for idx in range(betas.shape[0] - 1, -1, -1):
                model_input = {
                    "noisy_slice": denoised_slice,
                    "mel_spec": conditioned_mel_spec,
                    "time_step": time_series[idx],
                }

                c1 = 1 / (alphas[idx] ** 0.5)
                c2 = betas[idx] / (1 - alphas_cumprod[idx]) ** 0.5

                model_output = self.forward(model_input)

                denoised_slice = c1 * (
                    denoised_slice - c2 * model_output["noise_pred"].squeeze(1)
                )
                if idx > 0:
                    added_noise = torch.randn_like(denoised_slice)
                    sigma = (
                        (1.0 - alphas_cumprod[idx - 1])
                        / (1 - alphas_cumprod[idx])
                        * betas[idx]
                    ) ** 0.5
                    denoised_slice += sigma * added_noise
                denoised_slice = torch.clamp(denoised_slice, -1.0, 1.0)

                if show_process_plots:
                    sns.set_style("darkgrid")
                    ax = sns.lineplot(denoised_slice.squeeze(0).cpu().numpy())
                    ax.set_ylim((-1.0, 1.0))
                    plt.show()
            return {
                "denoised_slice": denoised_slice,
            }
