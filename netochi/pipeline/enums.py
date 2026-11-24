from enum import Enum


class InputGraph(Enum):
    MOSAIC = "mosaic"
    NOISY_MOSAIC = "noisy_mosaic"
    ERDOS_RENYI = "erdos_renyi"
    SWTA = "swta"
    SWTA_V2 = "swtaV2"