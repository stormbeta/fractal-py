import time
import qtoml
from typing import *
from dataclasses import dataclass, fields
import multiprocessing as mp
import math
import logging


class ConfigBase:
    def inline_copy(self, obj):
        for field in fields(self.__class__):
            setattr(self, field.name, getattr(obj, field.name))


# Config for multi-frame renders that isn't tied to values in config.toml
@dataclass
class MultiFrameConfig(ConfigBase):
    frame: int
    theta: float
    folder: str


@dataclass
class Config(ConfigBase):
    progress_indicator: bool
    save_render_data: bool
    save_histogram_png: bool

    global_resolution: int
    iteration_limit: int
    escape_threshold: float
    min_density: int
    max_density: int
    workers: int

    start: float
    stop: float
    frames: int

    log_level: str

    render_plane: Tuple[float, float, float, float]
    view_plane: Tuple[float, float, float, float]
    min_template: List[Union[str, float]]
    max_template: List[Union[str, float]]

    # Uncommon flags
    skip_hist_boundary_check: bool
    skip_hist_optimization: bool

    iteration_func: str

    def reload(self, config_file: Union[str, bytes]):
        self.inline_copy(Config.load(config_file))

    @classmethod
    def load(cls, config_file: Union[str, bytes] = 'config.toml'):
        cfg = cls(**{(k.name): None for k in fields(cls)})
        data: dict
        if isinstance(config_file, str):
            with open('config.toml', 'r') as fp:
                data = qtoml.load(fp)
        else:
            data = qtoml.loads(config_file.decode('utf-8'))
        cfg.iteration_func = data.get('iteration_func')
        cfg.theta = 0.0  # NOTE: Special var, should only be in-memory
        cfg.log_level = data.get('log_level', 'INFO')
        cfg._flags(data)
        cfg._multi(data)
        cfg.iteration_limit = pow(2, data['iteration_limit_power'])
        cfg.global_resolution = data['resolution']
        cfg.escape_threshold = data.get('escape_threshold', 2.0)
        density_range = data.get('density_range', [16, 16])
        cfg.min_density = density_range[0]
        cfg.max_density = density_range[1]
        cfg.render_plane = data.get('render_plane', [-2.0, -2.0, 2.0, 2.0])
        cfg.view_plane = data.get('view_plane', cfg.render_plane)
        cfg.min_template = data.get('m_min', [0.0, 0.0, "xmin", "ymin"])
        cfg.max_template = data.get('m_max', [0.0, 0.0, "xmax", "ymax"])
        workers = data.get('workers', -1)
        if workers > 0:
            cfg.workers = workers
        else:
            cfg.workers = workers + mp.cpu_count()
        assert cfg.min_density > 0 and cfg.max_density < 256
        assert cfg.iteration_limit <= 4096
        return cfg

    def _multi(self, data: dict) -> None:
        self.start = data.get('start', 0.0)
        self.stop = data.get('stop', 0.0)
        self.frames = data.get('frames', -1)

    def _flags(self, data: dict) -> None:
        self.skip_hist_boundary_check = data.get('skip_hist_boundary_check', False)
        self.skip_hist_optimization = data.get('skip_hist_optimization', False)
        self.progress_indicator = data.get('progress_indicator', True)
        self.save_render_data = data.get('save_render_data', True)
        self.save_histogram_png = data.get('save_histogram_png', False)

    def rshape(self) -> Tuple[int, int, int]:
        return (self.global_resolution, self.global_resolution, 3)

    def template_m_plane(self, theta: float = 0.0) -> Tuple[List[float], List[float]]:
        lookup = {'xmin': self.render_plane[0], 'ymin': self.render_plane[1],
                  'xmax': self.render_plane[2], 'ymax': self.render_plane[3],
                  'theta': theta}
        math_functions = {(func): getattr(math, func) for func in [func for func in dir(math) if not func.startswith('_')]}
        def template(value: Union[str, float]) -> float:
            if isinstance(value, str):
                return eval(value, math_functions, lookup)
            elif isinstance(value, float):
                return value
            else:
                raise TypeError(f"{value} is not a string or float!")
        return ([template(value) for value in self.min_template],
                [template(value) for value in self.max_template])


config = Config.load()
frame_params = MultiFrameConfig(-1, 0.0, "renders")
logging.basicConfig(level=config.log_level, format="%(message)s")
log = logging.getLogger(mp.current_process().name)
log.addHandler(logging.FileHandler('render.log', 'a'))


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
