#!/usr/bin/env python3

import math
import multiprocessing
import os
from datetime import datetime
from multiprocessing import Process

from .common import *

from cython cimport view

import numpy as np
from numpy import ndarray
cimport numpy as np
# cnp.import_array()

from cpython.mem cimport PyMem_Malloc, PyMem_Free
import cython
import array
from cpython cimport array
from libc.string cimport memcpy


render_outer = True
render_inner = False

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
"""

# TODO: Native inner loop
"""
Python is quite convenient overall, but the traces still feel awfully slow compared to what I expected.
I really don't want to write everything in C/C++, but perhaps we can use something like Cython to rewrite just the inner loop?
Worst-case, extract just the foundational logic and ability to write into a data file as it's own code + config file....
Though right now, that still makes up the bulk of the program

UPDATE: Much of the inner-most loop is now mostly native via Cython, resulting in over 4x speedup
        I still think I should try to do a pure C/C++ implementation though for comparison
        Or at least attempt to convert the remaining inner loop to pure Cython (currently uses numpy)
"""
cdef:
    struct Plane:
        double xmin, ymin, xmax, ymax

    struct Configuration:
        int max_iterations
        int trace_count

    # DEPRECATED
    struct Res:
        int width, height

    struct Coordinate:
        int x, y

    # [[ Zr, Zi ]
    #  [ Cr, Ci ]]
    struct Point4:
        double zr, zi, cr, ci

    # Yes I know numpy already implements all these, but the overhead for tiny 2x2 matrices is kind of large
    # Dot-product
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
        cdef:
            int resolution
            Plane plane
            double dx, dy

        def __cinit__(self, Plane plane, int resolution):
            self.resolution = resolution
            self.plane = plane
            self.dx = ((plane.xmax - plane.xmin) / resolution)
            self.dy = ((plane.ymax - plane.ymin) / resolution)

        # NOTE: Inlining these did basically nothing?
        cdef inline int x2column(self, double x):
            return <int>(((x - self.plane.xmin) / self.dx) - 1)

        cdef inline int y2row(self, double y):
            return <int>(self.resolution - int((y - self.plane.ymin) / self.dy) - 1)

        cdef inline double col2x(self, int x):
            return self.plane.xmin + (<double>x * self.dx)

        cdef inline double row2y(self, int y):
            return self.plane.ymax - (<double>y * self.dy)

    # TODO: Use this or get rid of it?
    #       numpy native arrays seem plenty fast to me, without the headaches
    class RenderData:
        cdef RenderWindow rwin
        cdef np.uint32_t* data
        # cdef np.ndarray[np.uint32] data

        def __cinit__(self, RenderWindow rwin, size_t buffer_size):
            # self.data = <np.uint32_t*> PyMem_Malloc(self.size * sizeof(np.uint32_t))
            self.data = <np.uint32_t*> PyMem_Malloc(buffer_size * sizeof(np.uint32_t))
            if not self.data:
                raise MemoryError()

        def __dealloc__(self):
            PyMem_Free(self.data)

    # Clamp value to nearest power of four
    cdef int squarepants(double value):
        return <int>math.pow(value, math.floor(math.log(value, 4)))

cdef class RenderConfig:
    cdef:
        RenderWindow rwin
        int iteration_limit
        Point4 r_min, r_max, r_diff, r_dt

    def __cinit__(self, RenderWindow rwin, int iteration_limit,
                  Point4 r_min, Point4 r_max):
        self.rwin = rwin
        self.iteration_limit = iteration_limit
        plane = rwin.plane
        self.r_min = r_min
        self.r_max = r_max
        # self.r_min = Point4(0.0, 0.0,
        #                     plane.xmin, plane.ymin)
        # self.r_max = Point4(0.0, 0.0,
        #                     plane.xmax, plane.ymax)
        self.r_diff = p4_sub(self.r_max, self.r_min)
        self.r_dt = p4_scalar_div(self.r_diff, rwin.resolution)


cdef render_histogram(RenderConfig rconfig, data: np.ndarray[np.uint32]):
    cdef:
        Plane plane = rconfig.rwin.plane
        RenderWindow rwin = rconfig.rwin
        # Point4 r_min = Point4(0.0, 0.0,
        #                       plane.xmin, plane.ymin)
        # Point4 r_max = Point4(0.0, 0.0,
        #                       plane.xmax, plane.ymax)
        # Point4 r_diff = p4_sub(r_max, r_min)
        # Point4 r_dt = p4_scalar_div(r_diff, rconfig.rwin.resolution)
        int i = 0
        Point4 point
    for x in range(rwin.resolution):
        for y in range(rwin.resolution):
            point = p4_add(rconfig.r_min, p4_dot(rconfig.r_dt, Point4(rwin.col2x(x), 0, 0, rwin.row2y(y))))
            # point = Point4(0, 0, rwin.col2x(x), rwin.row2y(y))
            for i in range(rconfig.iteration_limit):
                point = p4_iterate(point)
                # zr1 = point.zr * point.zr - point.zi * point.zi + point.cr
                # point.zi = 2 * point.zr * point.zi + point.ci
                # point.zr = zr1
                if point.zr * point.zr + point.zi * point.zi > 4:
                    # data[y*rwin.resolution + x] = i
                    data[x, y] = i
                    break
    # unclamped = np.sum(data)
    # data = np.power(4, np.maximum( np.floor(np.divide(np.log(data), math.log(4))), 2)).astype(dtype=np.uint32)
    data = np.minimum(np.maximum(data, 4), 16)
    # clamped = np.sum(data)
    # print(f"{unclamped} / {clamped} = {unclamped / clamped}")
    # print(f"Res: {clamped / rwin.resolution}")
    return data


# @cython.boundscheck(False)
# @cython.wraparound(False)
# @cython.overflowcheck(False)
@cython.infer_types(True)    # NOTE: Huge performance boost
@cython.cdivision(True)      # NOTE: Huge performance boost
def dothing(id: int, workers: int, angle: double):
    cdef Plane plane
    plane = Plane(-1.75, -1.25, 0.75, 1.25)
    traces = pow(2, 20)
    rwin = RenderWindow(plane, global_resolution)

    r_min = Point4(math.sin(-angle), math.cos(-angle),
                   plane.xmin, plane.ymin)
    r_max = Point4(math.sin(angle), math.cos(angle),
                   plane.xmax, plane.ymax)
    rconfig = RenderConfig(rwin, pow(2, 10), r_min, r_max)

    # TODO: avoid regenerating this for every worker
    histwin = RenderWindow(plane, rwin.resolution / 2)
    histconf = RenderConfig(histwin, pow(2, 8), r_min, r_max)
    histdata = np.full(fill_value=16, dtype=np.uint32, shape=(histwin.resolution, histwin.resolution))
    histdata = render_histogram(histconf, histdata)
    if id == 0:
        print(f"Traces: {math.log2(np.sum(histdata))}")
    rdata = np.full(fill_value=0, dtype=np.uint32, shape=(rwin.resolution, rwin.resolution * 3))
    render2(id, rconfig, histconf, histdata, rdata, workers, traces)


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.overflowcheck(False)
@cython.infer_types(True)    # NOTE: Huge performance boost
@cython.cdivision(True)      # NOTE: Huge performance boost
def render2(id: int,
            RenderConfig rconfig,
            RenderConfig histcfg,
            np.ndarray[np.uint32_t, ndim=2] histogram,
            np.ndarray[np.uint32_t, ndim=2] data,
            workers: int,
            traces: int):
    cdef:
        # const double density_factor = 0.5
        Plane plane = rconfig.rwin.plane
        RenderWindow rwin = rconfig.rwin
        int sqrt_chunks = histcfg.rwin.resolution
        double xpoints[65536]
        double ypoints[65536]
        # Loop vars
        int i, points, chunk_density
        Point4 p, chunk_pos, chunk_end, chunk_dt
        int chunks = sqrt_chunks*sqrt_chunks

    for chunk in range(id, chunks, workers):
        # if id == 0:
        #     print("=== NEW CHUNKER ===")
        chunk_col = chunk % sqrt_chunks
        chunk_row = math.floor(chunk / sqrt_chunks)
        chunk_density = histogram[chunk_col, chunk_row]
        # if id == 0:
        #     print(f"CHUNK {chunk}/{chunks}: {chunk_col}, {chunk_row} (traces: {chunk_density*chunk_density})")
        chunk_start = p4_add(rconfig.r_min,
                             p4_dot(histcfg.r_dt, Point4(chunk_col, 0, 0, chunk_row)))
        chunk_dt = p4_scalar_div(histcfg.r_dt, chunk_density)
        # if id == 0:
        # print(chunk_start)
        # print(f"chunk_dt: {chunk_dt}")
        # if id == 0:
        #     print(f"BANG: {a}, {b}")
        # data[a, b*3+1] = 128
        # chunk_start.zr = 0
        # chunk_start.zi = 0
        for s0 in range(chunk_density):
            for s1 in range(chunk_density):
                p = p4_add(chunk_start, p4_dot(chunk_dt, Point4(s0, 0, 0, s1)))
                # a = rwin.x2column(p.cr)
                # b = rwin.y2row(p.ci)
                # data[a, b*3 + 2] += 10
                escapes = False
                points = 0
                for i in range(rconfig.iteration_limit):
                    p = p4_iterate(p)
                    if p.zr * p.zr + p.zi * p.zi > 4:
                        escapes = True
                        break
                    # TODO: For shits and giggles, set this above 1. Yes I know that's completely wrong, don't care
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
    with open(f"render{id}.dat", "wb") as fp:
        fp.write(np.minimum(data, 255).tobytes())


# TODO: combine data in memory maybe? Though via disk has it's own advantages. Maybe generate RAM disk on the fly
# @cython.boundscheck(False)
# @cython.wraparound(False)
# @cython.overflowcheck(False)
# @cython.infer_types(True)    # NOTE: Huge performance boost
# @cython.cdivision(True)      # NOTE: Huge performance boost
# def render(id: int, resolution: Resolution, plane: Window, max_iter:int, workers: int, count: int, mandel):
# # def render(id: int,
# #            resolution: int,
# #            plane: Plane,
# #            iteration_limit: int,
# #            workers: int,
# #            traces: int,
# #            )
#     start_time = time.time()
#
#     rwin = RWindow(resolution, plane)
#     # TODO: There's got to be a better way to handle these conversions
#     cdef Res cres
#     cres.width = resolution.width
#     cres.height = resolution.height
#     cdef Plane cplane
#     cplane.xmin = plane.xmin
#     cplane.ymin = plane.ymin
#     cplane.xmax = plane.xmax
#     cplane.ymax = plane.ymax
#     rw = RenderData(cres, cplane, 3)
#
#     cdef int max_iterations = max_iter
#
#     # TODO: Needs to be revised under chunk scheme
#     # progress_increment = int(count/100)
#
#     # Ensure all numpy matrix operations are using cython
#     # NOTE: this doesn't actually seem to affect render times so much as static compile does
#     cdef np.ndarray[np.double_t, ndim=2] m_min, m_max, m_diff, m_inc, m_chunk_inc, chunk_pos, chunk_start
#
#     # Define the plane we wish to render in 4D space
#     # Well, sort of... I'm aware the math here isn't fully correct or even makes geometric sense
#     m_min = np.array([[0, 0], [plane.xmin, plane.ymin]])
#     m_max = np.array([[0, 0], [plane.xmax, plane.ymax]])
#     # m_min = np.array([rwin.xmin, rwin.ymax, rwin.xmax, rwin.ymax])
#     # m_max = np.array([rwin.xmax, rwin.ymin, rwin.xmin, rwin.ymin])
#     m_diff = m_max - m_min
#
#     # Loop vars - TODO: Not clear why these need to be declared up here...
#     cdef:
#         int xres = resolution.width
#         int yres = resolution.height
#         double xpoints[65536]
#         double ypoints[65536]
#         int limit = 0
#         double zr, zi, cr, ci
#         double zr1
#         Coordinate coord
#         int coord_x, coord_y
#         int baseptr
#         int escapes
#
#     # CHUNK CALCULATIONS
#     # TODO: This assumes square-shaped plane/resolution
#     sqrt_traces = int(math.sqrt(count))
#     chunks_per_side = math.floor(sqrt_traces / 128)
#     chunks = int(chunks_per_side * chunks_per_side)
#     cdef int sqrt_traces_per_chunk = int(sqrt_traces / chunks_per_side)
#
#     chunks_per_worker = math.ceil(chunks / workers)
#     if id ==0:
#         print(f"Chunks: {chunks}")
#         print(f"Traces per chunk: {count / chunks}")
#         print(f"Chunks per worker: {chunks_per_worker}")
#     m_inc  = (m_diff / sqrt_traces)
#     m_chunk_inc = (m_diff / chunks_per_side)
#
#     # z_min = 0 + 0j
#     # z_max = 0 + 0j
#     # c_min = rwin.xmin + 1j*rwin.ymin
#     # c_max = rwin.ymax + 1j*rwin.ymax
#     # TODO: Random pool is just the default way to handle nebula render
#     #       At higher counts, it would be better to actively increment points on the plane
#     #       The problem of course is unequal render times if handled naively
#     #       I think what I did originally was polar-coordinate staggered slices, but in hindsight
#     #       I don't think there's anything wrong with staggered cartesian slices
#     # rand_pool = np.random.random_sample(count*2+1)
#     # rand_pool_idx = 0
#     # dz_t = complex(m_max[0] - m_min[0] / subtraces, m_max[1] - m_min[1] / subtraces)
#     # dc_t = complex(m_max[2] - m_min[2] / subtraces, m_max[3] - m_min[3] / subtraces)
#     # dc_t = (m_min[3] - m_min[2]) / subtraces
#     # dz_t_chunk = dz_t * workers
#     # z0 = complex((m_min[0] + id*dz_t), (m_min[1] + id*dz_t))
#     # c0 = complex((m_min[2] + id*dc_t), (m_min[3] + id*dc_t))
#     # z0 = complex(m_min[0], m_min[1]) + id*dz_t
#     # c0 = complex(m_min[2], m_min[3]) + id*dc_t
#
#     #
#     # xtraces = int(count / 1)
#     # ytraces =
#     # # ytraces = int()
#     # current = [rwin.xmin, rwin.ymin]
#     # for s in range(count):
#     # z0 = 0 + 0j
#     progress_increment = (chunks / workers)/100
#     chunker = 0
#     cdef double rad_scale = 4.0 / math.pow(2, 32)
#     for chunk in range(id, chunks, workers):
#         chunk_col = chunk % chunks_per_side
#         chunk_row = math.floor(chunk / chunks_per_side)
#         chunk_start = m_min + np.dot(m_chunk_inc, [[chunk_col, 0], [0, chunk_row]])
#         # if id == 0:
#         #     print(f"CHUNK_POS: {chunk_row}, {chunk_col}")
#         #     print(f"CHUNK: {chunk}")
#         #     print(f"CHUNK_START: \n{chunk_start}")
#         if id == 0:
#             progress_milestone(start_time, int((chunker / chunks_per_worker) * 100))
#         chunker += 1
#         for s0 in range(sqrt_traces_per_chunk):
#             for s1 in range(sqrt_traces_per_chunk):
#                 # chunk_pos = chunk_start + np.dot([s0, s1], m_inc)
#                 # chunk_pos = chunk_start + np.dot([[s0, s1], [s0, s1]], m_inc)
#                 # TODO: Do this manually or figure out how to make numpy use cython directly
#                 chunk_pos = chunk_start + np.dot(m_inc, [[s0, 0], [0, s1]])
#                 # Only track process from one of the threads, as otherwise it just spams console output
#                 # z = 0 + 0j
#                 escapes = False
#                 # rand0, rand1 = rand_pool[rand_pool_idx], rand_pool[rand_pool_idx+1]
#                 # rand_pool_idx += 2
#                 # TODO: This math is definitely wrong, as it was supposed to render the normal buddhabrot if the z/c min/max were set normally
#                 #       Instead it renders some kind of abyssal horror
#                 # z = z_min.real + rand0 * z_diff.real + 1j*(z_min.imag+rand0*z_diff.imag)
#                 # c = c_min.real + rand1 * c_diff.real + 1j*(c_min.imag+rand1*c_diff.imag)
#                 # z = 0 + 0j
#                 # NOTE: Also wrong math, but it's interesting
#                 # z = m_min[0] + rand0 * m_diff[0].real + 1j*(m_min[1] + rand1 * m_diff[1])
#                 # c = m_min[2] + rand1 * m_diff[2].real + 1j*(m_min[3] + rand0 * m_diff[3])
#
#                 # Correct window math, at least more correct than anything else
#                 # z = m_min[0] + rand0 * m_diff[0].real + 1j*(m_min[1] + rand0 * m_diff[1])
#                 # c = m_min[2] + rand1 * m_diff[2].real + 1j*(m_min[3] + rand1 * m_diff[3])
#
#                 zr, zi = chunk_pos[0, :]
#                 cr, ci = chunk_pos[1, :]
#                 limit = 0
#
#                 for i in range(max_iterations):
#                     zr1 = zr*zr - zi*zi + cr
#                     zi = 2*zr*zi + ci
#                     zr = zr1
#                     if zr*zr + zi*zi > 4:
#                         escapes = True
#                         break
#                     limit += 1
#                     xpoints[i] = zr
#                     ypoints[i] = zi
#                 for i in range(limit):
#                     # coord = rw.plane2xy(xpoints[i], ypoints[i])
#                     coord_x = rw.x2column(xpoints[i])
#                     coord_y = rw.y2row(ypoints[i])
#                     # Ignore any points outside the render space
#                     if coord_x < 0 or coord_x >= xres or coord_y < 0 or coord_y >= yres:
#                         continue
#                     if escapes:
#                         # NOTE: Flipped x/y deliberately to rotate the image
#                         baseptr = coord_x*cres.width*3 + coord_y*3
#                         rw.data[baseptr] += 1
#                         rw.data[baseptr + 1] += i
#                         # TODO: add radius of initial point as final channel - may need to convert to double
#                         # rw.data[baseptr + 2] += ?
#                     # TODO: Consider bringing this option back in some form
#                     # elif render_inner:
#                     #     for i in range(limit):
#                     #         rwin.data[y, x * 3 + 2] += 1  # Non-escaping incrementor
#     # TODO: This is kind of messy - we should just use a numpy array to begin with
#     #       Also we should get rid of the rwindow.py code, most of it's unused now anyways
#     rwin.data = np.asarray(<np.uint32_t[:resolution.height, :(resolution.width*3)]> rw.data)
#     rwin.serialize(f"render{id}.dat")
#
#
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
#
# # def np_poly(arr0, arr1, inset, outset, maximum: int):
# #     arr1[:, outset::3] = ()
