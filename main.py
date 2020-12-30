#!/usr/bin/env python3

import png
from dataclasses import dataclass
from typing import *
import numpy
import multiprocessing
from multiprocessing import Process
import math


# TODO: Remove this and just use raw array access
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
        self.data = numpy.zeros(shape=(yres, xres*3), dtype=numpy.uint16)

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
            load = numpy.frombuffer(fp.read(), dtype=numpy.uint16)
            load.shape = (self.yres, 3*self.xres)
            self.data = load

    def iterator(self) -> Generator[int, int, Tuple[chr]]:
        for x in range(self.xres):
            for y in range(self.yres):
                yield x, y, self.get_pixel(x, y)


# TODO: Convert RWin to singleton for all values except data blob
width = 1024
height = 1024
plane = (-1.75, -1.25, 0.75, 1.25)
# plane = (-2.0, -2.0, 2.0, 2.0)


def render(dump: str, count: int):
    rwin = RWindow(width, height, plane)
    progress_increment = int(count/100)
    # Define the plane we wish to render in 4D space
    z_min = 0 + 0j
    z_max = 0 + 0j
    c_min = rwin.xmin + 1j*rwin.ymin
    c_max = rwin.ymax + 1j*rwin.ymax

    z_diff = z_max - z_min
    c_diff = c_max - c_min
    for s in range(count):
        # Only track process from one of the threads, as otherwise it just spams console output
        if(dump == 'render0.dat' and s % progress_increment == 0):
            print(f"Progress: {int(s/progress_increment)}%")
        # z = 0 + 0j
        points = []
        escapes = False
        # crand = numpy.random.random_sample(2)
        rand0, rand1 = numpy.random.random_sample(2)
        # TODO: This math is definitely wrong, as it was supposed to render the normal buddhabrot if the z/c min/max were set normally
        #       Instead it renders some kind of abyssal horror
        z = z_min.real + rand0 * z_diff.real + 1j*(z_min.imag+rand0*z_diff.imag)
        c = c_min.real + rand1 * c_diff.real + 1j*(c_min.imag+rand1*c_diff.imag)
        # c = ((rwin.xmax - rwin.xmin) * crand[0] + rwin.xmin) + 1j * ((rwin.ymax - rwin.ymin) * crand[1] + rwin.ymin)
        for i in range(200):
            z = z * z + c
            if z.real * z.real + z.imag * z.imag > 3:
                escapes = True
                break
            points.append((z.real, z.imag))
        if escapes:
            for p in points:
                r, i = p
                x, y = rwin.plane2xy(r, i)
                if x < 0 or x >= rwin.xres or y < 0 or y >= rwin.yres:
                    continue
                pixel = rwin.get_pixel(x, y)
                pixel.r += 1
                pixel.g += 1
                pixel.b += 1
                rwin.set_pixel(x, y, pixel)
    rwin.serialize(dump)


if __name__ == '__main__':
    workers = multiprocessing.cpu_count()

    # skip_render = True
    skip_render = False
    if not skip_render:
        traces = pow(2, 23)
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
    else:
        # Allow skipping render to play around with coloring/gradients without re-rendering every single time
        # NOTE: if you change plane/resolution, you must re-render!
        # TODO: Why do I get read-only here without double init?
        rwin0 = RWindow(width, height, plane)
        rwin0.deserialize("render.dat")
        rwin = RWindow(width, height, plane)
        rwin.data += rwin0.data


    # TODO: This is absurdly slow for some reason
    print("Normalizing...")
    rmax, gmax, bmax = 1, 1, 1

    # numpy is just as slow as plain iteration :(
    # with numpy.nditer(rwin.data, op_flags=['readwrite']) as it:
    # for rgb in numpy.nditer(rwin.data):
    #     if rgb % 3 == 0:
    #         rmax = max(rgb, rmax)
    #     elif rgb % 3 == 1:
    #         gmax = max(rgb, gmax)
    #     else:
    #         bmax = max(rgb, bmax)
    for x in range(rwin.xres):
        for y in range(rwin.yres):
            # TODO: HACKITY HACK - needed because current render.dat mistakenly always added origin, creating a massive outlier
            # if rwin.data[y][x * 3] > 50000:
            #     continue
            rmax = max(rwin.data[y][x * 3], rmax)
            gmax = max(rwin.data[y][x * 3 + 1], gmax)
            bmax = max(rwin.data[y][x * 3 + 2], bmax)

    floor = 0
    def logconvert(value: int, maximum: int):
        value = max(value - floor, 0)
        k = 255/math.log2(maximum)
        return k*math.log2(value+1)

    def sqrt_curve(value: int, maximum: int):
        return (255/math.sqrt(maximum)) * math.sqrt(value)

    def linear_curve(value: int, maximum: int):
        return value*(255/maximum)


    # with numpy.nditer(rwin.data, op_flags=['readwrite']) as it:
    #     for rgb in it:
    #         if rgb % 3 == 0:
    #             rgb[...] = logconvert(rgb, rmax)
    #         elif rgb % 3 == 1:
    #             rgb[...] = logconvert(rgb, gmax)
    #         else:
    #             rgb[...] = logconvert(rgb, bmax)
    print(f"red.max: {rmax}")
    print(f"green.max: {gmax}")
    print(f"blue.max: {bmax}")
    # Color normalization - different trace counts / resolution will produce different maximum scales
    for x in range(rwin.xres):
        for y in range(rwin.yres):
            red = rwin.data[y][x * 3]
            green = rwin.data[y][x * 3 + 1]
            blue = rwin.data[y][x * 3 + 2]

            # for j in range(2):
            #     rgb = rwin.data[y][x * 3 + j]
            #     rmx = rmax/3
            #     if rgb > rmx:
            #         # rgb = linear_curve(rmx, rmx) + sqrt_curve(rgb-rmx, rmax-rmx)
            #         # rgb_sqrt = sqrt_curve(rgb, rmax)
            #         # rgb_lin = linear_curve(rgb, rmax)
            #         rgb = min(linear_curve(rmx, rmax) + sqrt_curve(rgb, rmax), 255)
            #     else:
            #         rgb = min(linear_curve(rgb, rmx), 255)
            #     # rwin.data[y][x * 3 + j] = sqrt_curve(rgb, rmax)
            #     rwin.data[y][x * 3 + j] = rgb

            blue = logconvert(blue, rmax)
            green = (linear_curve(green, rmax) + sqrt_curve(green, rmax)) / 2
            red = (linear_curve(red, rmax) + sqrt_curve(red, rmax)) / 2

            # red = sqrt_curve(red, rmax)
            # green = sqrt_curve(green, gmax)
            # blue = sqrt_curve(blue, bmax)
            # red = linear_curve(red, rmax/90)
            # green = linear_curve(green, gmax/90)
            # blue = linear_curve(blue, bmax/90)

            rwin.data[y][x * 3] = red
            rwin.data[y][x * 3 + 1] = green
            rwin.data[y][x * 3 + 2] = blue

    print("Saving...")
    rwin.save("budda.png")
