from typing import Any, TYPE_CHECKING, Optional
from pytorch_lightning.utilities.types import STEP_OUTPUT
import torch
import torch.nn as nn

from models.base import BaseLightningModule
from utils.containers import (
    LearningParameters,
    Res1DDecoderParameters,
    MelSpecParameters,
)

if TYPE_CHECKING:
    from loss.aggregators import LossAggregator, LossOutput


class Res1DBlock(nn.Module):
    """
    1D Conv res block, similar to Jukebox paper. This is a try because the transformer one didn't
    regress to the wanted waveform, and the 1d vqvae doesn't reconstruct the sound well enough.
    """

    def __init__(
        self,
        num_channels: int,
        num_res_conv: int,
        dilation_factor: int,
        kernel_size: int,
    ) -> None:
        """
        Block initializer

        Args:
            num_channels (int): Number of block input channels
            num_res_conv (int): Number of convolution layers in a res block
            dilation_factor (int): Dilation exponential base for the res blocks
            kernel_size (int): Kernel size of each convolution in the res block
        """
        super().__init__()

        self.activation = nn.GELU

        # Create conv, activation, norm blocks
        self.res_block_modules = nn.ModuleList([])

        for idx in range(num_res_conv):
            # Keep output dimension equal to input dim
            dilation = dilation_factor**idx
            padding = (kernel_size + (kernel_size - 1) * (dilation - 1) - 1) // 2

            if idx != num_res_conv - 1:
                self.res_block_modules.append(
                    nn.Sequential(
                        nn.Conv1d(
                            num_channels,
                            num_channels,
                            kernel_size=kernel_size,
                            dilation=dilation,
                            padding=padding,
                        ),
                        self.activation(),
                        nn.BatchNorm1d(num_channels),
                    )
                )

            else:
                self.res_block_modules.append(
                    nn.Sequential(
                        nn.Conv1d(
                            num_channels,
                            num_channels,
                            kernel_size=kernel_size,
                            dilation=dilation,
                            padding=padding,
                        ),
                        self.activation(),
                    )
                )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward method of the res block

        Args:
            x (torch.Tensor): Tensor of size BS x C x W

        Returns:
            torch.Tensor: Tensor of size Bs x C x W
        """
        x_init = x.clone()
        for seq_module in self.res_block_modules:
            x = seq_module(x)

        return x + x_init


class Res1DBlockReverse(Res1DBlock):
    def __init__(
        self,
        num_channels: int,
        num_res_conv: int,
        dilation_factor: int,
        kernel_size: int,
    ) -> None:
        super().__init__(num_channels, num_res_conv, dilation_factor, kernel_size)

        # Create conv, activation, norm blocks
        self.res_block_modules = nn.ModuleList([])
        for idx in range(num_res_conv):
            # Keep output dimension equal to input dim
            dilation = dilation_factor ** (num_res_conv - idx - 1)
            padding = (kernel_size + (kernel_size - 1) * (dilation - 1) - 1) // 2

            if idx != num_res_conv - 1:
                self.res_block_modules.append(
                    nn.Sequential(
                        nn.Conv1d(
                            num_channels,
                            num_channels,
                            kernel_size=kernel_size,
                            dilation=dilation,
                            padding=padding,
                        ),
                        self.activation(),
                        nn.BatchNorm1d(num_channels),
                    )
                )

            else:
                self.res_block_modules.append(
                    nn.Sequential(
                        nn.Conv1d(
                            num_channels,
                            num_channels,
                            kernel_size=kernel_size,
                            dilation=dilation,
                            padding=padding,
                        ),
                        self.activation(),
                    )
                )


class Decoder1D(nn.Module):
    def __init__(
        self,
        decoder_parameters: Res1DDecoderParameters,
        mel_spec_parameters: MelSpecParameters,
        slice_length: int,
    ) -> None:
        """
        A simpler decoder than the old version, maybe will still need to push here some attention.
        """

        self.decoder_params = decoder_parameters

        dim_change_list = decoder_parameters.dim_change_sequence
        channel_list = [slice_length // (mel_spec_parameters.hop_length - 1)] + [
            decoder_parameters.hidden_size * (2 ** (idx + 1))
            for idx in reversed(range(len(dim_change_list)))
        ]

        super().__init__()
        if len(channel_list) != len(dim_change_list) + 1:
            raise ValueError(
                "The channel list length must be greater than the dimension change list by 1"
            )
        self.activation = nn.GELU()

        # Create the module lists for the architecture
        self.end_conv = nn.Conv1d(
            channel_list[-1],
            decoder_parameters.input_channels,
            kernel_size=3,
            padding=1,
        )
        self.conv_list = nn.ModuleList(
            [
                Res1DBlockReverse(
                    channel_list[idx],
                    decoder_parameters.num_res_block_conv,
                    decoder_parameters.dilation_factor,
                    decoder_parameters.kernel_size,
                )
                for idx in range(len(dim_change_list))
            ]
        )
        if decoder_parameters.dim_add_kernel_add % 2 != 0:
            raise ValueError("dim_add_kernel_size must be an even number.")

        self.dim_change_module_list = nn.ModuleList(
            [
                nn.ConvTranspose1d(
                    channel_list[idx],
                    channel_list[idx + 1],
                    kernel_size=dim_change_list[idx]
                    + decoder_parameters.dim_add_kernel_add,
                    stride=dim_change_list[idx],
                    padding=decoder_parameters.dim_add_kernel_add // 2,
                )
                for idx in range(len(dim_change_list))
            ]
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        for _, (conv, dim_change) in enumerate(
            zip(self.conv_list, self.dim_change_module_list)
        ):
            z = conv(z)
            z = dim_change(z)
            z = self.activation(z)

        x_out = self.end_conv(z)

        return x_out


class Res1DLightningModule(BaseLightningModule):
    def __init__(
        self,
        base_model: Decoder1D,
        learning_params: LearningParameters,
        loss_aggregator: Optional["LossAggregator"] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Any = None,
        **kwargs,
    ) -> None:
        super().__init__(
            base_model, learning_params, loss_aggregator, optimizer, scheduler, **kwargs
        )
        if not optimizer:
            self.optimizer = torch.optim.AdamW(
                self.parameters(),
                lr=learning_params.learning_rate,
                weight_decay=learning_params.weight_decay,
            )

    def forward(self, x: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        reconstructed_slice = self.model(x["mel_spec"])
        return {"slice": reconstructed_slice}

    def training_step(
        self, batch: dict[str, torch.Tensor], batch_idx: int
    ) -> STEP_OUTPUT:
        # Ensure there is loss and an optimizer during training
        if not self.loss_aggregator:
            raise ValueError("Must include a loss aggregator during training")
        if not self.optimizer:
            raise ValueError("Must include an optimizer during training")

        batch["mel_spec"] = batch["mel_spec"].squeeze(1)
        outputs = self.forward(batch)
        loss = self.loss_aggregator.compute(outputs, batch)
        self._log_losses("training", loss)
        return loss.total

    def validation_step(
        self, batch: dict[str, torch.Tensor], batch_idx: int
    ) -> STEP_OUTPUT | None:
        batch["mel_spec"] = batch["mel_spec"].squeeze(1)
        outputs = self.forward(batch)
        if self.loss_aggregator is not None:
            loss = self.loss_aggregator.compute(outputs, batch)
            self._log_losses("validation", loss)

    def _log_losses(self, phase_name: str, loss: "LossOutput") -> None:
        for loss_key, loss_value in loss.individuals.items():
            self.log(f"{phase_name}_{loss_key}", loss_value)

        self.log(f"{phase_name}_total_loss", loss.total, prog_bar=True)
