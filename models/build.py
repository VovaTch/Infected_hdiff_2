from typing import Dict

from utils.containers import MelSpecParameters
from .mel_spec_converters import MelSpecConverter, MEL_SPEC_CONVERTERS


def build_mel_spec_converter(
    type: str, mel_spec_params: MelSpecParameters
) -> MelSpecConverter:
    assert type in MEL_SPEC_CONVERTERS, f"Unknown converter type {type}"
    mel_spec_converter = MEL_SPEC_CONVERTERS[type](mel_spec_params)
    return mel_spec_converter
