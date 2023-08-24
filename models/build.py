from typing import Dict

from models.base import MelSpecConverter
from utils.containers import MelSpecParameters
from .base import MelSpecConverter, IMelSpecConverterFactory
import models.mel_spec_converters as mel_spec_converters

MEL_SPEC_CONVERTERS: Dict[str, MelSpecConverter] = {
    "simple": mel_spec_converters.SimpleMelSpecConverter,
    "scaled_image": mel_spec_converters.ScaledImageMelSpecConverter,
}


class MelSpecConverterFactory(IMelSpecConverterFactory):
    @staticmethod
    def build_mel_spec_converter(
        type: str, mel_spec_params: MelSpecParameters
    ) -> MelSpecConverter:
        assert type in MEL_SPEC_CONVERTERS, f"Unknown converter type {type}"
        mel_spec_converter = MEL_SPEC_CONVERTERS[type](mel_spec_params)
        return mel_spec_converter
