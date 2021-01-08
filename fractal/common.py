import time
from typing import *

from dataclasses import dataclass, fields


@dataclass()
class ConfigSingleton:
    def reset(self, obj: 'ConfigSingleton'):
        for field in fields(self.__class__):
            setattr(self, field.name, getattr(obj, field.name))


@dataclass()
class Flags(ConfigSingleton):
    progress_indicator: bool = True
    save_render_data: bool = True
    save_histogram_png: bool = False


@dataclass()
class Config(ConfigSingleton):
    global_resolution: int = 1024

    def rshape(self) -> Tuple[int, int, int]:
        return (self.global_resolution, self.global_resolution, 3)


flags = Flags()
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
