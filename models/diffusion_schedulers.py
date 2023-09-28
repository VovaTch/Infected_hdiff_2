import torch
import torch.nn.functional as F
import numpy as np

from models.base import DiffusionScheduler


def _cosine_schedule(timesteps: int, s=0.008, power: float = 2) -> torch.Tensor:
    t_running = torch.linspace(1, timesteps, timesteps)
    alpha_running_sqrt = torch.cos(np.pi / 2 * (t_running / timesteps + s) / (1 + s))
    return torch.pow(alpha_running_sqrt, power)


class CosineScheduler:
    betas: torch.Tensor
    alphas: torch.Tensor
    alphas_cumprod: torch.Tensor
    alphas_cumprod_prev: torch.Tensor
    sqrt_recip_alphas: torch.Tensor
    sqrt_alphas_cumprod: torch.Tensor
    sqrt_one_minus_alphas_cumprod: torch.Tensor
    posterior_variance: torch.Tensor

    def __init__(self, num_steps: int, device: str = "cpu") -> None:
        f_function = _cosine_schedule(num_steps).to(device)

        # Pre-calculate different terms for closed form
        self.alphas_cumprod = f_function.to(device)
        self.alphas_cumprod_prev = F.pad(
            self.alphas_cumprod[:-1], (1, 0), value=1.0
        ).to(device)
        self.betas = 1 - torch.div(self.alphas_cumprod, self.alphas_cumprod_prev)
        self.betas[self.betas > 0.99] = 0.99

        self.alphas = 1.0 - self.betas
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        self.posterior_variance = (
            self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )


def _linear_beta_schedule(timesteps, start=1e-4, end=0.05):
    return torch.linspace(start, end, timesteps)


class LinearScheduler:
    betas: torch.Tensor
    alphas: torch.Tensor
    alphas_cumprod: torch.Tensor
    alphas_cumprod_prev: torch.Tensor
    sqrt_recip_alphas: torch.Tensor
    sqrt_alphas_cumprod: torch.Tensor
    sqrt_one_minus_alphas_cumprod: torch.Tensor
    posterior_variance: torch.Tensor

    def __init__(self, num_steps: int, device: str = "cpu") -> None:
        self.betas = _linear_beta_schedule(num_steps).to(device)

        # Pre-calculate different terms for closed form
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        self.posterior_variance = (
            self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )


DIFFUSION_SCHEDULERS: dict[str, type[DiffusionScheduler]] = {
    "cosine": CosineScheduler,
    "linear": LinearScheduler,
}
