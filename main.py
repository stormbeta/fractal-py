#!/usr/bin/env python3

import math
import multiprocessing
import os
import time
from collections import namedtuple
from datetime import datetime
from multiprocessing import Process
from struct import pack, unpack, calcsize
from typing import *

import numpy as np
import png

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
        self.xmin = plane[0]
        self.ymin = plane[1]
        self.xmax = plane[2]
        self.ymax = plane[3]
        self.res = resolution
        self.win = window

        self.dx = ((self.win.xmax - self.win.xmin) / self.res.width)
        self.dy = ((self.win.ymax - self.win.ymin) / self.res.height)
        if self.dx != self.dy:
            print("WARNING: Aspect ratio mismatch!")
        self.data = np.zeros(shape=(resolution.height, resolution.width*3), dtype=np.uint32)

    def xy2plane(self, x: int, y: int) -> Tuple[float, float]:
        return (self.xmin + (x * self.dx),
                self.ymax - y * self.dy)

    def plane2xy(self, x: float, y: float) -> Tuple[int, int]:
        return (int((x - self.xmin) / self.dx) - 1,
                self.yres - int((y - self.ymin) / self.dy) - 1)

    def save(self, name: str):
        with open(name, "wb") as fp:
            writer = png.Writer(self.xres, self.yres, greyscale=False)
            writer.write(fp, self.data.astype('uint8'))

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


# TODO: Convert RWin to singleton for all values except data blob
# max_iter = 1000
max_iter = 100
res = Resolution(1024, 1024)
# res = Resolution(2048, 2048)
plane = Window(-1.75, -1.25, 0.75, 1.25)
# plane = Window(-2.0, -2.0, 2.0, 2.0)


def render(dump: str, count: int):
    start_time = time.time()

    rwin = RWindow(res, plane)
    progress_increment = int(count/100)

    # Define the plane we wish to render in 4D space
    # Well, sort of... I'm aware the math here isn't fully correct or even makes geometric sense
    # m_min = np.array([0, 0, rwin.xmin, rwin.ymin])
    # m_max = np.array([0, 0, rwin.xmax, rwin.ymax])
    m_min = np.array([rwin.xmin, rwin.ymax, rwin.xmax, rwin.ymax])
    m_max = np.array([rwin.xmax, rwin.ymin, rwin.xmin, rwin.ymin])

    m_diff = m_max - m_min
    # z_min = 0 + 0j
    # z_max = 0 + 0j
    # c_min = rwin.xmin + 1j*rwin.ymin
    # c_max = rwin.ymax + 1j*rwin.ymax
    randi = np.random.random_sample(count*4+1)
    r_idx = 0
    for s in range(count):
        # Only track process from one of the threads, as otherwise it just spams console output
        if(dump == 'render0.dat' and s % progress_increment == 0):
            print(f"Progress: {int(s/progress_increment)}%")
        # z = 0 + 0j
        escapes = False
        # rand0, rand1 = randi[r_idx], randi[r_idx+1]
        rand0, rand1, rand2, rand3 = randi[r_idx], randi[r_idx+1], randi[r_idx+2], randi[r_idx+3]
        r_idx += 4
        # TODO: This math is definitely wrong, as it was supposed to render the normal buddhabrot if the z/c min/max were set normally
        #       Instead it renders some kind of abyssal horror
        # z = z_min.real + rand0 * z_diff.real + 1j*(z_min.imag+rand0*z_diff.imag)
        # c = c_min.real + rand1 * c_diff.real + 1j*(c_min.imag+rand1*c_diff.imag)
        # z = 0 + 0j
        z = m_min[0] + rand0 * m_diff[0].real + 1j*(m_min[1] + rand1 * m_diff[1])
        c = m_min[2] + rand1 * m_diff[2].real + 1j*(m_min[3] + rand0 * m_diff[3])
        # m_min[0] + rand0
        # c = ((rwin.xmax - rwin.xmin) * crand[0] + rwin.xmin) + 1j * ((rwin.ymax - rwin.ymin) * crand[1] + rwin.ymin)
        xpoints = np.zeros(max_iter, dtype=float)
        ypoints = np.zeros(max_iter, dtype=float)
        limit = 0
        for i in range(max_iter):
            z = z * z + c
            if z.real * z.real + z.imag * z.imag > 4:
                escapes = True
                break
            limit += 1
            xpoints[i] = z.real
            ypoints[i] = z.imag
        for i in range(limit):
            # Weirdly, using numpy here is actually slower than calling rwin.plane2xy
            # Guessing it's due to the relatively small size of the points lists
            # npx = (xpoints - rwin.xmin) / rwin.dx - 1
            # npy = rwin.yres - ((ypoints - rwin.ymin)/rwin.dy) - 1
            # x = int(npx[i])
            # y = int(npy[i])
            x, y = rwin.plane2xy(xpoints[i], ypoints[i])
            # Ignore any points outside the render space
            if x < 0 or x >= rwin.xres or y < 0 or y >= rwin.yres:
                continue
            if escapes:
                rwin.data[y, x*3] += 1      # Baseline incrementor
                rwin.data[y, x*3 + 1] += i  # Based on iteration count to reach this point
            # TODO: Leave disabled for now, adds _WAY_ too much render time to do both inner/outer traces
            #       unless actually intending to use both, even for relatively small trace/iteration counts
            # else:
            #     for i in range(limit):
            #         rwin.data[y, x * 3 + 2] += 1  # Non-escaping incrementor
    if (dump == 'render0.dat'):
        print(f'{dump} running time is {start_time - time.time():.2f} s')
    rwin.serialize(dump)


def np_log_curve(arr0, arr1, inset, outset, maximum: int):
    k = 255/math.log2(maximum)
    arr1[:, outset::3] = np.multiply(k, np.log2(arr0[:, inset::3]))

def np_sqrt_curve(arr0, arr1, inset, outset, maximum: int):
    k = 255/math.sqrt(maximum)
    arr1[:, outset::3] = k * np.sqrt(arr0[:, inset::3])

def np_quasi_curve(arr0, arr1, inset, outset, maximum: int):
    linear_k = 255/maximum
    sqrt_k = 255/math.sqrt(maximum)
    arr1[:, outset::3] = (sqrt_k*np.sqrt(arr0[:, inset::3]) + linear_k*arr0[:, inset::3]) / 2

def np_linear(arr0, arr1, inset, outset, maximum: int):
    arr1[:, outset::3] = (255/maximum) * arr0[:, inset::3]


if __name__ == '__main__':
    workers = multiprocessing.cpu_count() - 1

    # TODO: CLI interface / args / flags
    skip_render = True
    # skip_render = False

    if skip_render:
        # Allow skipping render to play around with coloring/gradients without re-rendering every single time
        # NOTE: Safe to use on render data from different plane/resolution, as that's stored alongside point data
        rwin = RWindow.deserialize(f"render.dat")
        output = RWindow(resolution=rwin.res, window=rwin.win)
    else:
        traces = pow(2, 21)
        processes = []
        traces_per_process = int(traces / workers)
        print(f"Traces per process: {traces_per_process}")
        for i in range(workers):
            proc = Process(target=render, args=(f"render{i}.dat", int(traces / workers)))
            processes.append(proc)
            proc.start()
        for proc in processes:
            proc.join()

        # MERGE
        rwin = RWindow(resolution=res, window=plane)
        print(f"Merging {workers} datasets...")
        for i in range(workers):
            pdat = RWindow.deserialize(f"render{i}.dat")
            rwin.data += pdat.data
        rwin.serialize("render.dat")
        [os.remove(f"render{i}.dat") for i in range(workers)]
        output = RWindow(resolution=res, window=plane)

    print("Colorizing...")

    # TODO Convert this to array so that we can combine max + channel offset?
    # Native numpy is ludicrously faster than iterating manually in python
    nmax = np.max(rwin.data[:, 0::3])
    itermax = np.max(rwin.data[:, 1::3])
    innermax = np.max(rwin.data[:, 2::3])
    print(f"nmax: {nmax}")
    print(f"itermax: {itermax}")
    print(f"innermax: {innermax}")

    #         red = (linear_curve(red, rmax) + sqrt_curve(red, rmax)) / 2
    # rwin.data[:, 0::3] = (255/rmax) * rwin.data[:, 0::3]
    np_sqrt_curve(rwin.data, output.data, 0, 0, nmax/3)
    # np_log_curve(rwin.data, 0, rmax*3)
    np_sqrt_curve(rwin.data, output.data, 1, 1, itermax)
    np_log_curve(rwin.data, output.data, 0, 2, nmax*1.5)

    # np_linear(rwin.data, 0, rmax/7)
    # np_linear(rwin.data, 1, gmax/7)
    # np_log_curve(rwin.data, 2, bmax*3)

    # Clamp values
    output.data = np.minimum(output.data, 255)

    print("Saving...")
    # TODO: Include resolution/iteration count in filename?
    output.save(f"renders/nebula_{int(datetime.now().timestamp())}.png")
