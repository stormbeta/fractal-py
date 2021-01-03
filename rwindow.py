from collections import namedtuple
from struct import calcsize, pack, unpack
from typing import *

import numpy as np


Resolution: Type[Tuple[int, int]] = namedtuple('Resolution', ['width', 'height'])
Point: Type[Tuple[float, float]] = namedtuple('Point', ['x', 'y'])
Window: Type[Tuple[float, float, float, float]] = namedtuple('Window', ['xmin', 'ymin', 'xmax', 'ymax'])


class RWindow:
    res: Resolution
    win: Window
    # xres: int
    # yres: int
    # xmin: float
    # xmax: float
    # ymin: float
    # ymax: float

    # TODO: This allows non-1-to-1 mapping of aspect ratio between plane and resolution
    #       A) Auto-correct either x or y to match plane
    #       B) Only have plane + resolution
    # def __init__(self, xres: int, yres: int, plane: Tuple[float, float, float, float]):
    def __init__(self, resolution: Resolution, window: Window):
        # TODO: Fix remaining references
        self.xres = resolution.width
        self.yres = resolution.height
        self.xmin = window[0]
        self.ymin = window[1]
        self.xmax = window[2]
        self.ymax = window[3]
        self.res = resolution
        self.win = window

        self.dx = ((self.win.xmax - self.win.xmin) / self.res.width)
        self.dy = ((self.win.ymax - self.win.ymin) / self.res.height)
        if self.dx != self.dy:
            print("WARNING: Aspect ratio mismatch!")
        # TODO: dtype should probably be a global constant
        #       Ultra resolutions could get large, may want to consider zipping or cutting channel count (100MP 32-bit = 400MB)
        self.data = np.zeros(shape=(resolution.height, resolution.width*3), dtype=np.uint32)

    def xy2plane(self, x: int, y: int) -> Tuple[float, float]:
        return (self.xmin + (x * self.dx),
                self.ymax - y * self.dy)

    def plane2xy(self, x: float, y: float) -> Tuple[int, int]:
        return (int((x - self.xmin) / self.dx) - 1,
                self.yres - int((y - self.ymin) / self.dy) - 1)

    def serialize(self, name: str):
        with open(name, "wb") as fp:
            fp.write(pack('iidddd', *self.res, *self.win))
            fp.write(self.data.tobytes())

    # TODO: This creates RWindow as read-only
    @classmethod
    def deserialize(cls, name: str) -> 'RWindow':
        with open(name, "rb") as fp:
            resolution = Resolution(*unpack('ii', fp.read(calcsize('ii'))))
            window = Window(*unpack('dddd', fp.read(calcsize('dddd'))))
            load = np.frombuffer(fp.read(), dtype=np.uint32)
            load.shape = (resolution.height, 3*resolution.width)
        rwin = cls(resolution=resolution, window=window)
        rwin.data = load
        return rwin

