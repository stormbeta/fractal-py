import logging
import math
import multiprocessing as mp
import time
from dataclasses import dataclass, fields
from functools import wraps
from typing import *

import qtoml


class ConfigBase:
    def inline_copy(self, obj):
        for field in fields(self.__class__):
            setattr(self, field.name, getattr(obj, field.name))


# Config for multi-frame renders that isn't tied to values in config.toml
@dataclass
class FrameConfig(ConfigBase):
    frame: int
    theta: float
    folder: str


# TODO: Ideally this should be serializable back to config, but that'd be an annoying amount of work at this point
# TODO: Should probably create categories within the config file but don't really want to break compatibility
@dataclass
class Config(ConfigBase):
    progress_indicator: bool
    save_render_data: bool
    save_histogram_png: bool
    enable_gui: bool

    global_resolution: int
    iteration_limit: int
    escape_threshold: float
    min_density: int
    max_density: int
    workers: int

    start: float
    stop: float
    frames: int
    framestep: str

    log_level: str

    render_plane: Tuple[float, float, float, float]
    view_plane: Tuple[float, float, float, float]
    min_template: List[Union[str, float]]
    max_template: List[Union[str, float]]

    # Uncommon flags
    skip_hist_boundary_check: bool
    skip_hist_optimization: bool

    iteration_sig: str
    iteration_func: str

    color_scale_max: int
    color_scale: List[List[float]]
    color_algo: List[str]

    def reload(self, config_file: Union[str, bytes]):
        self.inline_copy(Config.load(config_file))

    @classmethod
    def load(cls, config_file: Union[str, bytes] = 'config.toml'):
        cfg = cls(**{(k.name): None for k in fields(cls)})
        data: dict
        if isinstance(config_file, str):
            with open(config_file, 'r') as fp:
                data = qtoml.load(fp)
        else:
            data = qtoml.loads(config_file.decode('utf-8'))
        cfg.iteration_func = data.get('iteration_func')
        cfg.iteration_sig = data.get('iteration_sig')
        cfg.global_resolution = data['resolution']
        cfg.theta = 0.0  # NOTE: Special var, should only be in-memory
        cfg.log_level = data.get('log_level', 'INFO')
        cfg._flags(data)
        cfg._multi(data)
        cfg._color(data)
        cfg.iteration_limit = pow(2, data['iteration_limit_power'])
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
        self.framestep = data.get('framestep', 'linear')

    def _color(self, data: dict) -> None:
        self.color_scale = data.get('color_scale')
        self.color_algo = data.get('color_algo', ['sqrt_curve', 'sqrt_curve', 'sqrt_curve'])
        # self.color_scale_max = math.sqrt(self.global_resolution) * 4
        self.color_scale_max = 256

    def _flags(self, data: dict) -> None:
        self.skip_hist_boundary_check = data.get('skip_hist_boundary_check', False)
        self.skip_hist_optimization = data.get('skip_hist_optimization', False)
        self.progress_indicator = data.get('progress_indicator', True)
        self.save_render_data = data.get('save_render_data', True)
        self.save_histogram_png = data.get('save_histogram_png', False)
        self.enable_gui = data.get('enable_gui', False)

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
frame_params = FrameConfig(-1, config.start, "renders")
logging.basicConfig(level=config.log_level, format="[%(name)s]: %(message)s")
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


def debug_timer(func: Callable) -> Callable:
    @wraps(func)
    def wrapped(*args, **kwargs):
        start_time = time.time()
        ret_value = func(*args, **kwargs)
        log.info(f"{func.__name__}: {seconds_convert(time.time() - start_time)}")
        return ret_value
    return wrapped