#!/usr/bin/env python3

import png
from dataclasses import dataclass
from typing import *
import numpy
import multiprocessing
from multiprocessing import Process
import math


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
    pixel_max: Pixel

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
        # fudge_factor = 2
        self.pixel_max = Pixel(255*1.4, 255*2.2, 255*3.2)

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

    def normalize(self):
        for x in range(self.xres):
            for y in range(self.yres):
                pixel = self.get_pixel(x, y)
                pixel.r = pixel.r * (255/self.pixel_max.r)
                pixel.g = pixel.g * (255/self.pixel_max.g)
                pixel.b = pixel.b * (255/self.pixel_max.b)
                self.set_pixel(x, y, pixel)

    def iterator(self) -> Generator[int, int, Tuple[chr]]:
        for x in range(self.xres):
            for y in range(self.yres):
                yield x, y, self.get_pixel(x, y)


# TODO: Convert RWin to singleton for all values except data blob
width = 1024
height = 1024
# plane = (-2.0, -1.5, 1.0, 1.5)
plane = (-2.0, -2.0, 2.0, 2.0)


def render(dump: str, count: int):
    rwin = RWindow(width, height, plane)
    progress_increment = int(count/100)
    for s in range(count):
        if(dump == 'render0.dat' and s % progress_increment == 0):
            print(f"{dump}: {int(s/progress_increment)}%")
        z = 0 + 0j
        points = []
        escapes = False
        crand = numpy.random.random_sample(2)
        rwin.xmax - rwin.xmin

        c = ((rwin.xmax - rwin.xmin) * crand[0] + rwin.xmin) + 1j * ((rwin.ymax - rwin.ymin) * crand[1] + rwin.ymin)
        for i in range(100):
            z = z * z + c
            if z.real * z.real + z.imag * z.imag > 4:
                escapes = True
                break
        if escapes:
            for p in points:
                r, i = p
                x, y = rwin.plane2xy(r, i)
                if x < 0 or x > rwin.xres or y < 0 or y > rwin.yres:
                    continue
                pixel = rwin.get_pixel(x, y)
                pixel.r += 1
                pixel.g += 2
                pixel.b += 3
                rwin.set_pixel(x, y, pixel)
    rwin.serialize(dump)


if __name__ == '__main__':
    pcount = multiprocessing.cpu_count()

    # skip_render = True
    skip_render = False
    if not skip_render:
        traces = pow(2, 21)
        processes = []
        traces_per_process = int(traces / pcount)
        print(f"Traces per process: {traces_per_process}")
        for i in range(pcount):
            proc = Process(target=render, args=(f"render{i}.dat", int(traces / pcount)))
            processes.append(proc)
            proc.start()
        for proc in processes:
            proc.join()

    # MERGE
    rwin = RWindow(width, height, plane)
    pdata = []
    for i in range(pcount):
        print(f"Merging {i+1}/{pcount}")
        pdat = RWindow(width, height, plane)
        pdat.deserialize(f"render{i}.dat")
        rwin.data += pdat.data

    # TODO: This is absurdly slow for some reason
    print("Normalizing...")
    def logconvert(value: int, maximum: int):
        k = 255/math.log2(maximum)
        return k*math.log2(value+1)


    rmax, gmax, bmax = 1, 1, 1

    # for x in range(rwin.xres):
    #     for y in range(rwin.yres):
    #         rmax = max(rwin.data[y][x * 3], rmax)
    #         gmax = max(rwin.data[y][x * 3 + 1], gmax)
    #         bmax = max(rwin.data[y][x * 3 + 2], bmax)
    # for x in range(rwin.xres):
    #     for y in range(rwin.yres):
    #         rwin.data[y][x * 3] = logconvert(rwin.data[y][x * 3], rmax*2)
    #         # rwin.data[y][x * 3] *= 255/rmax
    #         rwin.data[y][x * 3 + 1] = logconvert(rwin.data[y][x * 3 + 1], gmax*2)
    #         rwin.data[y][x * 3 + 2] = logconvert(rwin.data[y][x * 3 + 2], bmax*2)
    print("Saving...")
    rwin.save("budda.png")








    # if __name__ == '__main__':
    #
    #     # for x, y, pixel in rwin.iterator():
    #     #     pixel.r = x
    #     #     pixel.g = y
    #     #     pixel.b = x*y/2
    #     #     rwin.set_pixel(x, y, pixel)
    #     #
    #     # sys.exit(0)
    #
    #     for s in range(100000):
    #         z = 0 + 0j
    #         points = []
    #         escapes = False
    #         crand = numpy.random.random_sample(2)
    #         c = 4*(crand[0]-0.5) + 4j*(crand[1]-0.5)
    #         for i in range(50):
    #             points.append((z.real, z.imag))
    #             z = z*z + c
    #             if z.real*z.real + z.imag*z.imag > 4:
    #                 escapes = True
    #                 break
    #         if escapes:
    #             for p in points:
    #                 x, y = rwin.plane2xy(c.real, c.imag)
    #                 pixel = rwin.get_pixel(x, y)
    #                 pixel.r += 10
    #                 rwin.set_pixel(x, y, pixel)
    #
    #     # for x in range(rwin.xres):
    #     #     for y in range(rwin.yres):
    #     #         cr, ci = rwin.xy2plane(x, y)
    #     #         zr = 0.0
    #     #         zi = 0.0
    #     #         z = 0 + 0j
    #     #         c = cr + ci*1j
    #     #         for i in range(50):
    #     #             zr0 = zr
    #     #             zi0 = zi
    #     #             z = z*z + c
    #     #             # zr = (zr*zr - zi*zi) + cr
    #     #             # zi = (2*zr0*zi) + ci
    #     #             # if (zr*zr + zi*zi) > 4:
    #     #             if z.imag*z.imag + z.real*z.real > 4:
    #     #                 # px, py = rwin.plane2xy(cr, ci)
    #     #                 rwin.set_pixel(x, y, Pixel(80.0*(2.0-(math.sqrt(cr*cr + ci*ci))), i*3, i*5))
    #     #                 break
    #     rwin.save("buddhabrot.png")
    #
    # # See PyCharm help at https://www.jetbrains.com/help/pycharm/
