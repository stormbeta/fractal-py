import time
from typing import *


# TODO: these should be in some kind of config singleton instead
global_resolution: int = 1024
# global_resolution: int = 2048
enable_progress_indicator: bool = False
enable_histogram_render: bool = True
enable_render_dat_save: bool = True
enable_histogram_powerclamp: bool = True
RSHAPE = (global_resolution, global_resolution * 3)


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
