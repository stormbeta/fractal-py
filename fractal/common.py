import time
import toml
from typing import *
from dataclasses import dataclass, fields


@dataclass
class Config:
    progress_indicator: bool
    save_render_data: bool
    save_histogram_png: bool

    global_resolution: int
    iteration_limit: int
    escape_threshold: float
    min_density: int
    max_density: int

    # Uncommon flags
    skip_hist_boundary_check: bool
    skip_hist_optimization: bool

    def inline_copy(self, obj):
        for field in fields(self.__class__):
            setattr(self, field.name, getattr(obj, field.name))

    @classmethod
    def load(cls):
        cfg = cls(**{(k.name): None for k in fields(cls)})
        with open('config.toml', 'r') as fp:
            data = toml.load(fp)
            cfg.iteration_limit = pow(2, data['iteration_limit_power'])
            cfg.global_resolution = data['resolution']
            cfg.progress_indicator = data.get('progress_indicator', True)
            cfg.save_render_data = data.get('save_render_data', True)
            cfg.save_histogram_png = data.get('save_histogram_png', False)
            cfg.escape_threshold = data.get('escape_threshold', 2.0)
            density_range = data.get('density_range', [16, 16])
            cfg.min_density = density_range[0]
            cfg.max_density = density_range[1]
            cfg.skip_hist_boundary_check = data.get('skip_hist_boundary_check', False)
            cfg.skip_hist_optimization = data.get('skip_hist_optimization', False)

            assert cfg.min_density > 0 and cfg.max_density < 256
            assert cfg.iteration_limit <= 65536
        return cfg

    def rshape(self) -> Tuple[int, int, int]:
        return (self.global_resolution, self.global_resolution, 3)


config = Config.load()


def seconds_convert(seconds: Union[float, int]) -> str:
    return "{:02d}:{:02d}:{:02d}".format(
        int(seconds / 3600),
        int(seconds % 3600 / 60),
        int(seconds) % 60)


def progress_milestone(start_time: float, percent: float) -> None:
    if percent > 0:
        eta = seconds_convert(int((100 - percent) * (time.time() - start_time) / percent))
    else:
        eta = "?"
    print(f"\rProgress: {percent:.2f}% (ETA: {eta} ¯\\_(ツ)_/¯)", end='')
