import time
import json
from typing import *

from dataclasses import dataclass, fields

# From https://pypi.org/project/singleton-decorator
def singleton(cls):
    instances = {}
    def wrapper(*args, **kwargs):
        if cls not in instances:
          instances[cls] = cls(*args, **kwargs)
        return instances[cls]
    return wrapper


# @dataclass
# class ConfigSingleton:
#     def reset(self, obj: 'ConfigSingleton'):
#         for field in fields(self.__class__):
#             setattr(self, field.name, getattr(obj, field.name))


# @singleton
class Config:
    progress_indicator: bool = True
    save_render_data: bool = True
    save_histogram_png: bool = False

    global_resolution: int = 1024
    iteration_limit: int = pow(2, 10)

    def __init__(self):
        with open('config.json', 'r') as fp:
            data = json.load(fp)
            self.iteration_limit = pow(2, data['iteration_limit_power'])
            self.global_resolution = data['resolution']
            self.progress_indicator = data.get('progress_indicator', self.progress_indicator)
            self.save_render_data = data.get('save_render_data', self.save_render_data)
            self.save_histogram_png = data.get('save_histogram_png', self.save_histogram_png)


    def rshape(self) -> Tuple[int, int, int]:
        return (self.global_resolution, self.global_resolution, 3)


config = Config()


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
