import time
from typing import *

import png

from .rwindow import Resolution

def write_png(data, resolution: Resolution, name: str):
    with open(name, "wb") as fp:
        writer = png.Writer(resolution.width, resolution.height, greyscale=False)
        writer.write(fp, data.astype('uint8'))


def seconds_convert(seconds: Union[float, int]) -> str:
    return "{:02d}:{:02d}:{:02d}".format(
        int(seconds / 3600),
        int(seconds % 3600 / 60),
        int(seconds) % 60)


def progress_milestone(start_time: float, percent: int) -> None:
    if percent > 0:
        eta = seconds_convert(int((100 - percent) * (time.time() - start_time) / percent))
    else:
        eta = "?"
    print(f"\rProgress: {percent}% (ETA: {eta})", end='')
