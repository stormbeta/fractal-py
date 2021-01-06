#!/usr/bin/env python3

import math
import multiprocessing as mp
from datetime import datetime
import png

cimport numpy as np
import cython
import numpy as np
from cpython.mem cimport PyMem_Malloc, PyMem_Free

from .common import *

# TODO: user interface
"""
At some point this really needs a GUI for more active exploration of various parameters, especially coloring
Even automated CLI isn't enough for something like color that can displayed in real time along continuous axes
"""

# TODO: CUDA / CPU?
"""
It's unclear if GPU compute could even help much here, except maybe in specific scenarios like combined inner/outer
rendering. We'd need to batch operations across multiple traces at once, but with no way to easily intervene for points that escape or not
"""

# TODO: Monte Carlo Importance Sampling
"""
It's pretty obvious the most interesting traces come from the points near the mandelbrot set boundary, so ideally we could
cluster traces near said boundary to drastically improve rendering efficiency.
I don't think this is practical to do systematically, but it seems like there are algorithms for random sampling based on
an importance map, which could be pre-generated as the standard mandelbrot set for the given resolution

UPDATE: more or less done via histogram density trick, but it looks like this makes the iteration count coloring almost worthless
        Not sure why, but iteration count coloring is now mostly uniform.
"""

cdef:
    # Yes I know numpy already implements all these, but the overhead for tiny 2x2 matrices is large, numpy is meant for larger datasets
    # Dot-product
    # [[ Zr, Zi ]   \/  [[ Zr, Zi ]
    #  [ Cr, Ci ]]  /\   [ Cr, Ci ]]
    cdef Point4 p4_dot(Point4 a, Point4 b):
        return Point4(a.zr * b.zr + a.zi * b.cr, a.zr * b.zi + a.zi * b.ci,
                      a.cr * b.zr + a.ci * b.cr, a.cr * b.zi + a.ci * b.ci)

    cdef Point4 p4_scalar_mult(Point4 a, double scalar):
        return Point4(a.zr * scalar, a.zi * scalar,
                      a.cr * scalar, a.ci * scalar)

    cdef Point4 p4_scalar_div(Point4 a, double divisor):
        return Point4(a.zr / divisor, a.zi / divisor,
                      a.cr / divisor, a.ci / divisor)

    cdef Point4 p4_add(Point4 a, Point4 b):
        return Point4(a.zr + b.zr, a.zi + b.zi,
                      a.cr + b.cr, a.ci + b.ci)

    cdef Point4 p4_sub(Point4 a, Point4 b):
        return Point4(a.zr - b.zr, a.zi - b.zi,
                      a.cr - b.cr, a.ci - b.ci)

    cdef Point4 p4_iterate(Point4 a):
        return Point4(a.zr * a.zr - a.zi * a.zi + a.cr, 2 * a.zr * a.zi + a.ci,
                      a.cr                            , a.ci)

    class RenderWindow:
        def __cinit__(self, Plane plane, int resolution):
            self.resolution = resolution
            self.plane = plane
            self.dx = ((plane.xmax - plane.xmin) / resolution)
            self.dy = ((plane.ymax - plane.ymin) / resolution)

        # NOTE: Inlining these doesn't seem to actually help performance
        cdef int x2column(self, double x):
            return <int>(((x - self.plane.xmin) / self.dx) - 1)

        cdef int y2row(self, double y):
            return <int>(self.resolution - int((y - self.plane.ymin) / self.dy) - 1)

        cdef double col2x(self, int x):
            return self.plane.xmin + (<double>x * self.dx)

        cdef double row2y(self, int y):
            return self.plane.ymax - (<double>y * self.dy)


    class RenderConfig:
        def __cinit__(self, RenderWindow rwin, int iteration_limit,
                      Point4 m_min, Point4 m_max):
            self.rwin = rwin
            self.iteration_limit = iteration_limit
            plane = rwin.plane
            self.m_min = m_min
            self.m_max = m_max
            self.m_diff = p4_sub(self.m_max, self.m_min)
            self.m_dt = p4_scalar_div(self.m_diff, rwin.resolution)


# TODO: Configurable clamping of values to powers of 2
"""
If you don't clamp density to powers of two, you wind up with rounding artifacts that are pretty visible
in static renders, though not so much if rendering animated sequences
"""
cdef np.ndarray[np.uint8_t, ndim=2] render_histogram(RenderConfig histcfg, int min_density, int max_density):
    cdef:
        Plane plane = histcfg.rwin.plane
        RenderWindow rwin = histcfg.rwin
        np.ndarray[np.uint8_t, ndim=2] data
        double hist_k
        int i = 0
        int x, y
        Point4 point

    data = np.full(fill_value=min_density, dtype=np.uint8, shape=(rwin.resolution, rwin.resolution))
    if min_density == max_density:
        # print("Skipping histogram render as density interval is constant")
        return data

    for x in range(rwin.resolution):
        for y in range(rwin.resolution):
            point = p4_add(histcfg.m_min, p4_dot(histcfg.m_dt, Point4(x, 0, 0, y)))
            for i in range(histcfg.iteration_limit):
                point = p4_iterate(point)
                if point.zr * point.zr + point.zi * point.zi > 4:
                    data[x, y] = i
                    break

    # Linearly scale histogram to fit min/max density
    # NOTE: minor artifacting if density is not a power of 2 TODO: power of 2, or just cleanly divisible by 2?
    hist_k = max_density / np.max(data)
    # return np.maximum(np.multiply(hist_k, data), min_density).astype(np.uint8)
    return np.maximum(data, min_density).astype(np.uint8)


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.overflowcheck(False)
@cython.infer_types(True)    # NOTE: Huge performance boost
@cython.cdivision(True)      # NOTE: Huge performance boost
def nebula(id: int, shared_data: mp.Array, workers: int, dt: double):
    cdef:
        Plane plane
        np.ndarray[np.uint8_t, ndim=2] histdata
    # TODO: This needs to be global or something
    plane = Plane(-1.75, -1.25, 0.75, 1.25)
    # plane = Plane(-2, -2, 2, 2)
    rwin = RenderWindow(plane, global_resolution)

    # TODO: This should be more configurable - it determines the primary location of the render after all
    # IMPORTANT: histogram and main render *must* use the same plane, m_min, and m_max!
    # m_min = Point4(math.sin(-time_var), math.cos(-time_var),
    #                plane.xmin, plane.ymin)
    # m_max = Point4(math.sin(time_var), math.cos(time_var),
    #                plane.xmax, plane.ymax)
    m_min = Point4(1-math.cos(dt), math.sin(dt), plane.xmin, plane.ymin)
    m_max = Point4(1-math.cos(dt/2), math.sin(dt/2), plane.xmax, plane.ymax)
    # Standard mandelbrot plane, where z0 is always 0 + 0i
    # m_min = Point4(0.0, 0.0, plane.xmin, plane.ymin)
    # m_max = Point4(0.0, 0.0, plane.xmax, plane.ymax)

    rconfig = RenderConfig(rwin, pow(2, 10), m_min, m_max)

    # TODO: avoid regenerating histogram for every worker? Though honestly unless I parallelize it doesn't really matter
    #       and it's fast enough anyways other than per-frame, and per-frame I probably only want one worker per frame anyways
    # TODO: Give some way of dynamically setting desired trace instead of byproduct of density/histogram?
    #       This is kind of hard to calculate in reverse
    # traces = pow(2, 20)

    # Minimum traces per side of any given chunk (traces per chunk equals density^2)
    # If equal, disable histogram optimization
    cdef int min_density = 16
    cdef int max_density = 16
    assert min_density > 0

    # Render histogram of mandelbrot set, and linearly scale density down to a controllable max
    histwin = RenderWindow(plane, rwin.resolution)
    histcfg = RenderConfig(histwin, pow(2, 8), m_min, m_max)
    histdata = render_histogram(histcfg, min_density, max_density)
    if enable_histogram_powerclamp:
        histdata[...] = 2 * np.floor_divide(histdata, 2)
    if id == 0:
        print(f"log2(traces) = {math.log2(np.sum(histdata)):.2f}")
        print(f"density interval: ({min_density}, {max_density})")
        if enable_histogram_render:
            output_filename = f"histogram/histogram{int(datetime.now().timestamp())}.png"
            with open(output_filename, "wb") as fp:
                writer = png.Writer(histwin.resolution, histwin.resolution, greyscale=True)
                writer.write(fp, histdata.astype('uint8'))

    rdata = np.full(fill_value=0, dtype=np.float64, shape=RSHAPE)
    render2(id, rconfig, histcfg, histdata, rdata, workers)
    with shared_data.get_lock():
        shared = np.frombuffer(shared_data.get_obj(), dtype=np.float64)
        shared.shape = RSHAPE
        shared += rdata


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.overflowcheck(False)
@cython.infer_types(True)    # NOTE: Big performance boost
@cython.cdivision(True)      # NOTE: Big performance boost
def render2(id: int,
            RenderConfig rconfig,
            RenderConfig histcfg,
            np.ndarray[np.uint8_t, ndim=2] histogram,
            np.ndarray[np.float64_t, ndim=2] data,
            workers: int):

    # NOTE: iteration_limit _must_ be less than or equal to 65536 because of the static xpoints/ypoints arrays
    #       it provides a noticeable speedup over dynamic allocation, and animated renders shouldn't have high iteration counts anyways
    assert rconfig.iteration_limit <= 65536

    cdef:
        # const double density_factor = 0.5
        Plane plane = rconfig.rwin.plane
        RenderWindow rwin = rconfig.rwin
        int sqrt_chunks = histcfg.rwin.resolution
        double xpoints[65536]
        double ypoints[65536]

        # Loop vars
        int i, points, chunk_density, chunk_col, chunk_row
        Point4 p, chunk_start, chunk_pos, chunk_end, chunk_dt
        int chunks = sqrt_chunks*sqrt_chunks
        int count = 0
        double radius

    if id == 0 and enable_progress_indicator:
        progress_total = np.sum(np.reshape(np.copy(histogram), newshape=(pow(histcfg.rwin.resolution, 2),) )[id::workers])

    start_time = time.time()

    # Core rendering loops
    for chunk in range(id, chunks, workers):
        chunk_col = chunk % sqrt_chunks
        chunk_row = math.floor(chunk / sqrt_chunks)
        chunk_density = histogram[chunk_col, chunk_row]
        assert chunk_density > 0
        if id == 0 and enable_progress_indicator:
            # print(f"CHUNKER {chunk}/{chunks}: {chunk_col}, {chunk_row} (traces: {chunk_density*chunk_density})")
            count += chunk_density
            if chunk % 32 == 0:
                progress_milestone(start_time, ((count / progress_total) * 100))
        chunk_start = p4_add(rconfig.m_min,
                             p4_dot(histcfg.m_dt, Point4(chunk_col, 0, 0, chunk_row)))
        chunk_dt = p4_scalar_div(histcfg.m_dt, chunk_density)
        for s0 in range(chunk_density):
            for s1 in range(chunk_density):
                p = p4_add(chunk_start, p4_dot(chunk_dt, Point4(s0, 0, 0, s1)))
                radius = math.sqrt(p.zr * p.zr + p.zi * p.zi)
                escapes = False
                points = 0
                for i in range(rconfig.iteration_limit):
                    p = p4_iterate(p)
                    if p.zr * p.zr + p.zi * p.zi > 4:
                        escapes = True
                        break
                    # TODO: For shits and giggles, set this above 1. Yes I know that's completely wrong
                    points += 1
                    xpoints[i] = p.zr
                    ypoints[i] = p.zi
                if escapes:
                    for i in range(points):
                        x, y = xpoints[i], ypoints[i]
                        if plane.xmin < x < plane.xmax and plane.ymin < y < plane.ymax:
                            a, b = rwin.x2column(x), rwin.y2row(y)
                            data[a, b * 3] += 1
                            data[a, b * 3 + 1] += i
                            data[a, b * 3 + 2] += radius


# TODO: split these into separate file
def np_log_curve(arr0, arr1, inset, outset, maximum: int):
    k = 255/math.log2(maximum)
    arr1[:, outset::3] = np.multiply(k, np.log2(arr0[:, inset::3]))

def np_sqrt_curve(arr0, arr1, inset, outset, maximum: int):
    k = 255/math.sqrt(maximum)
    arr1[:, outset::3] = k * np.sqrt(arr0[:, inset::3])

def np_inv_sqrt_curve(arr0, arr1, inset, outset, maximum: int):
    k = 255/math.sqrt(maximum)
    arr1[:, outset::3] = maximum - k * np.sqrt(arr0[:, inset::3])

def np_quasi_curve(arr0, arr1, inset, outset, maximum: int):
    linear_k = 255/maximum
    sqrt_k = 255/math.sqrt(maximum)
    arr1[:, outset::3] = (sqrt_k*np.sqrt(arr0[:, inset::3]) + linear_k*arr0[:, inset::3]) / 2

def np_linear(arr0, arr1, inset, outset, maximum: int):
    arr1[:, outset::3] = (255/maximum) * arr0[:, inset::3]
