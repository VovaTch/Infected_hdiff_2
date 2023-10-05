import torch
import torch.nn.functional as F
import torchaudio


def create_slices_from_file(
    mp3_file_path: str, sampling_rate: int, slice_length: int
) -> torch.Tensor:
    long_slice, music_sampling_rate = torchaudio.load(mp3_file_path, format="mp3")  # type: ignore
    long_slice = resample_if_necessary(long_slice, sampling_rate, music_sampling_rate)
    long_slice = mix_down_if_necessary(long_slice)
    long_slice = right_pad_if_necessary(long_slice, slice_length)
    return long_slice.view((-1, 1, slice_length)).half()


def resample_if_necessary(
    signal: torch.Tensor, desired_sampling_rate: int, music_sampling_rate: int
) -> torch.Tensor:
    """
    Function to change the sampling rate of a music track

    Args:
        signal (torch.Tensor): Music slice, expects `C x L` size
        sampling_rate (int): Sampling rate, default for MP3 is 44100.

    Returns:
        torch.Tensor: Resampled signal
    """
    if desired_sampling_rate != music_sampling_rate:
        resampler = torchaudio.transforms.Resample(
            desired_sampling_rate, music_sampling_rate
        )
        signal = resampler(signal)
    return signal


def mix_down_if_necessary(signal: torch.Tensor) -> torch.Tensor:
    """
    Function to merge down music channels, currently the code doesn't support more than 1 channel.

    Args:
        signal (torch.Tensor): Signal to be mixed down, shape `C x L`

    Returns:
        torch.Tensor: Mixed-down signal, shape `1 x L`
    """

    if signal.shape[0] > 1:
        signal = torch.mean(signal, dim=0, keepdim=True)
    return signal


def right_pad_if_necessary(signal: torch.Tensor, slice_length: int) -> torch.Tensor:
    """
    Function aimed to keep all the slices at a constant size, pad with 0 if the slice is too short.

    Args:
        signal (torch.Tensor): Input slice, shape `1 x L*`

    Returns:
        torch.Tensor: Output slice, shape `1 x L` padded with zeroes.
    """
    length_signal = signal.shape[1]
    if length_signal % slice_length != 0:
        num_missing_samples = slice_length - length_signal % slice_length
        last_dim_padding = (0, num_missing_samples)
        signal = F.pad(signal, last_dim_padding)
    return signal
