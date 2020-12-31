#!/usr/bin/env python3

import png
from dataclasses import dataclass
from typing import *
import numpy as np
import multiprocessing
from multiprocessing import Process
import math

import functools
import time

# decorator to measure running time
def measure_running_time(echo=True):
    def decorator(func):
        @functools.wraps(func)
        def wrapped(*args, **kwargs):
            t_1 = time.time()
            ans = func(*args, **kwargs)
            t_2 = time.time()
            if echo:
                print(f'{func.__name__}() running time is {t_2 - t_1:.2f} s')
            return ans
        return wrapped
    return decorator


# TODO: Remove this and just use raw array access?
@dataclass
class Pixel:
    r: int
    g: int
    b: int


class RWindow:
    xres: int
    yres: int
    xmin: float
    xmax: float
    ymin: float
    ymax: float

    # TODO: This allows non-1-to-1 mapping of aspect ratio between plane and resolution
    #       A) Auto-correct either x or y to match plane
    #       B) Only have plane + resolution
    def __init__(self, xres: int, yres: int, plane: Tuple[float, float, float, float]):
        self.xres = xres
        self.yres = yres
        self.xmin = plane[0]
        self.ymin = plane[1]
        self.xmax = plane[2]
        self.ymax = plane[3]

        self.dx = ((self.xmax - self.xmin) / self.xres)
        self.dy = ((self.ymax - self.ymin) / self.yres)
        if self.dx != self.dy:
            print("WARNING: Aspect ratio mismatch!")
        self.data = np.zeros(shape=(yres, xres*3), dtype=np.uint16)

    def set_pixel(self, x: int, y: int, pixel: Pixel):
        self.data[y][x*3] = pixel.r
        self.data[y][x*3+1] = pixel.g
        self.data[y][x*3+2] = pixel.b

    def get_pixel(self, x: int, y: int) -> Pixel:
        return Pixel(self.data[y][x*3],
                     self.data[y][x*3+1],
                     self.data[y][x*3+2])

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
            fp.write(self.data.tobytes())

    # WARNING: Will overwrite existing self.data!
    def deserialize(self, name: str):
        with open(name, "rb") as fp:
            load = np.frombuffer(fp.read(), dtype=np.uint16)
            load.shape = (self.yres, 3*self.xres)
            self.data = load

    def iterator(self) -> Generator[int, int, Tuple[chr]]:
        for x in range(self.xres):
            for y in range(self.yres):
                yield x, y, self.get_pixel(x, y)


# TODO: Convert RWin to singleton for all values except data blob
# width = 2048
# height = 2048
# max_iter = 1000
max_iter = 100
width = 1024
height = 1024
plane = (-1.75, -1.25, 0.75, 1.25)
# plane = (-2.0, -2.0, 2.0, 2.0)


def render(dump: str, count: int):
    start_time = time.time()

    rwin = RWindow(width, height, plane)
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
            # points.append((z.real, z.imag))
        if not escapes:
            # Weirdly, using numpy here is actually slower than calling rwin.plane2xy
            # Guessing it's due to the relatively small size of the points lists
            # npx = (xpoints - rwin.xmin) / rwin.dx - 1
            # npy = rwin.yres - ((ypoints - rwin.ymin)/rwin.dy) - 1
            for i in range(limit):
                x, y = rwin.plane2xy(xpoints[i], ypoints[i])
                # x = int(npx[i])
                # y = int(npy[i])
                if x < 0 or x >= rwin.xres or y < 0 or y >= rwin.yres:
                    continue
                # TODO: This is kind of silly - we should decouple color channels from render data
                #       That way render data channels can focus on information only available at render time
                #       All colorization calculations are already deferred to post-merge anyways
                rwin.data[y, x*3] += 1
                rwin.data[y, x*3 + 1] += 1
                rwin.data[y, x*3 + 2] += 1
    if (dump == 'render0.dat'):
        print(f'{dump} running time is {start_time - time.time():.2f} s')
    rwin.serialize(dump)


if __name__ == '__main__':
    workers = multiprocessing.cpu_count()

    # skip_render = True
    skip_render = False
    if not skip_render:
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
        rwin = RWindow(width, height, plane)
        for i in range(workers):
            print(f"Merging {i+1}/{workers}")
            pdat = RWindow(width, height, plane)
            pdat.deserialize(f"render{i}.dat")
            rwin.data += pdat.data
        rwin.serialize("render.dat")
        # TODO: Delete unmerged data
    else:
        # Allow skipping render to play around with coloring/gradients without re-rendering every single time
        # NOTE: if you change plane/resolution, you must re-render!
        # TODO: Why do I get read-only here without double init?
        rwin0 = RWindow(width, height, plane)
        rwin0.deserialize("render.dat")
        rwin = RWindow(width, height, plane)
        rwin.data += rwin0.data


    print("Normalizing...")
    rmax, gmax, bmax = 1, 1, 1

    # Native numpy is ludicrously faster than iterating in python
    rmax = np.max(rwin.data[:, 0::3])
    gmax = np.max(rwin.data[:, 1::3])
    bmax = np.max(rwin.data[:, 2::3])

    def np_log_curve(arr, offset, maximum: int):
        k = 255/math.log2(maximum)
        arr[:, offset::3] = np.multiply(k, np.log2(arr[:, offset::3]))

    def np_sqrt_curve(arr, offset, maximum: int):
        k = 255/math.sqrt(maximum)
        arr[:, offset::3] = k * np.sqrt(arr[:, offset::3])

    #         red = (linear_curve(red, rmax) + sqrt_curve(red, rmax)) / 2
    def np_quasi_curve(arr, offset, maximum: int):
        linear_k = 255/maximum
        sqrt_k = 255/math.sqrt(maximum)
        arr[:, offset::3] = (sqrt_k*np.sqrt(arr[:, offset::3]) + linear_k*arr[:, offset::3]) / 2

    def np_linear(arr, offset, maximum: int):
        arr[:, offset::3] = (255/maximum) * arr[:, offset::3]


    # rwin.data[:, 0::3] = (255/rmax) * rwin.data[:, 0::3]
    np_sqrt_curve(rwin.data, 0, rmax/3)
    # np_log_curve(rwin.data, 0, rmax*3)
    np_sqrt_curve(rwin.data, 1, gmax/5)
    np_log_curve(rwin.data, 2, bmax*1.5)

    # np_linear(rwin.data, 0, rmax/7)
    # np_linear(rwin.data, 1, gmax/7)
    # np_log_curve(rwin.data, 2, bmax*3)
    rwin.data = np.minimum(rwin.data, 255)

    floor = 0
    def log_curve(value: int, maximum: int):
        value = max(value - floor, 0)
        k = 255/math.log2(maximum)
        return k*math.log2(value+1)

    def sqrt_curve(value: int, maximum: int):
        return (255/math.sqrt(maximum)) * math.sqrt(value)

    def linear_curve(value: int, maximum: int):
        return value*(255/maximum)

    print(f"red.max: {rmax}")
    print(f"green.max: {gmax}")
    print(f"blue.max: {bmax}")

    # TODO: This is absurdly slow for some reason
    # Color normalization - different trace counts / resolution will produce different maximum scales
    if False:
        for x in range(rwin.xres):
            for y in range(rwin.yres):
                red = rwin.data[y][x * 3]
                green = rwin.data[y][x * 3 + 1]
                blue = rwin.data[y][x * 3 + 2]
                blue = log_curve(blue, rmax)
                green = sqrt_curve(green, gmax)
                # green = (linear_curve(green, rmax) + sqrt_curve(green, rmax)) / 2
                # red = (linear_curve(red, rmax) + sqrt_curve(red, rmax)) / 2
                # red = sqrt_curve()
                red = linear_curve(red, rmax)
                rwin.data[y][x * 3] = red
                rwin.data[y][x * 3 + 1] = green
                rwin.data[y][x * 3 + 2] = blue

    print("Saving...")
    rwin.save("budda.png")
