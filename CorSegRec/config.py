from dataclasses import dataclass

@dataclass
class DPCConfig:
    patch_sizes: tuple = (15, 7)
    patch_resize_to: int = 11
    omega: float = 5.0
    max_steps: int = 500
    stop_distance: float = 1.5
    neighbor_mode: str = "6"
    spacing: tuple = (1.0, 1.0, 1.0)

DEFAULT_CONFIG = DPCConfig()
