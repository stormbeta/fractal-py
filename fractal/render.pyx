#!/usr/bin/env python3

import math
import multiprocessing
import os
from datetime import datetime
from multiprocessing import Process

from .common import *
from .rwindow import Resolution, Window, RWindow

import numpy as np
from numpy import ndarray
cimport numpy as np
# cnp.import_array()

from cpython.mem cimport PyMem_Malloc, PyMem_Free
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

# TODO: Native inner loop
"""
Python is quite convenient overall, but the traces still feel awfully slow compared to what I expected.
I really don't want to write everything in C/C++, but perhaps we can use something like Cython to rewrite just the inner loop?
Worst-case, extract just the foundational logic and ability to write into a data file as it's own code + config file....
Though right now, that still makes up the bulk of the program
"""
cdef:
    struct Plane:
        double xmin, ymin, xmax, ymax

    struct Res:
        int width, height

    struct Coordinate:
        int x, y

    struct Point:
        double x, y

    class RenderData:
        cdef Res resolution
        cdef Plane plane
        cdef np.int32_t* data
        cdef double dx, dy
        cdef size_t size

        def __cinit__(self, Res res, Plane plane):
            self.size = res.height * res.width * 3
            self.data = <np.int32_t*> PyMem_Malloc(self.size * sizeof(np.int32_t))
            self.resolution = res
            self.plane = plane
            self.dx = ((self.plane.xmax - self.plane.xmin) / self.resolution.width)
            self.dy = ((self.plane.ymax - self.plane.ymin) / self.resolution.height)
            if not self.data:
                raise MemoryError()

        def plane2xy(self, double x, double y):
            cdef Coordinate result
            cdef int rx = <int>(((x - self.plane.xmin) / self.dx) - 1)
            cdef int ry = <int>(self.resolution.height - int((y - self.plane.ymin) / self.dy) - 1)
            result.x = rx
            result.y = ry
            return result

        def __dealloc__(self):
            PyMem_Free(self.data)


# TODO: combine data in memory maybe? Though via disk has it's own advantages. Maybe generate RAM disk on the fly
def render(id: int, resolution: Resolution, plane: Window, max_iter:int, workers: int, count: int):
    start_time = time.time()

    rwin = RWindow(resolution, plane)
    # TODO: There's got to be a better way to handle these conversions
    cdef Res cres
    cres.width = resolution.width
    cres.height = resolution.height
    cdef Plane cplane
    cplane.xmin = plane.xmin
    cplane.ymin = plane.ymin
    cplane.xmax = plane.xmax
    cplane.ymax = plane.ymax
    rw = RenderData(cres, cplane)

    # TODO: Needs to be revised under chunk scheme
    # progress_increment = int(count/100)

    # Define the plane we wish to render in 4D space
    # Well, sort of... I'm aware the math here isn't fully correct or even makes geometric sense
    m_min = np.array([[0, 0], [plane.xmin, plane.ymin]])
    m_max = np.array([[0, 0], [plane.xmax, plane.ymax]])
    # m_min = np.array([rwin.xmin, rwin.ymax, rwin.xmax, rwin.ymax])
    # m_max = np.array([rwin.xmax, rwin.ymin, rwin.xmin, rwin.ymin])
    m_diff = m_max - m_min

    cdef int xres = resolution.width
    cdef int yres = resolution.height

    # Loop vars - TODO: Not clear why these need to be declared up here...
    cdef double xpoints[16384]
    cdef double ypoints[16384]
    cdef int limit = 0
    cdef:
        double zr, zi, cr, ci
        double zr1
        Coordinate coord

    # CHUNK CALCULATIONS
    # TODO: This assumes square-shaped plane/resolution
    sqrt_traces = int(math.sqrt(count))
    chunks_per_side = math.floor(sqrt_traces / 128)
    chunks = int(chunks_per_side * chunks_per_side)
    sqrt_traces_per_chunk = int(sqrt_traces / chunks_per_side)
    chunks_per_worker = math.ceil(chunks / workers)
    if id ==0:
        print(f"Chunks: {chunks}")
        print(f"Traces per chunk: {count / chunks}")
        print(f"Chunks per worker: {chunks_per_worker}")
    m_inc  = (m_diff / sqrt_traces)
    m_chunk_inc = (m_diff / chunks_per_side)

    # if id == 8:
    #     print(m_min)
    #     print(m_max)
    #     print(f"Minc:\n{m_inc}")
    #     print(f"Mchunk_inc:\n{m_chunk_inc}")

    # z_min = 0 + 0j
    # z_max = 0 + 0j
    # c_min = rwin.xmin + 1j*rwin.ymin
    # c_max = rwin.ymax + 1j*rwin.ymax
    # TODO: Random pool is just the default way to handle nebula render
    #       At higher counts, it would be better to actively increment points on the plane
    #       The problem of course is unequal render times if handled naively
    #       I think what I did originally was polar-coordinate staggered slices, but in hindsight
    #       I don't think there's anything wrong with staggered cartesian slices
    # rand_pool = np.random.random_sample(count*2+1)
    # rand_pool_idx = 0
    # TODO: This is a shitty way of dividing work up, but w/e
    # subtraces = int(math.sqrt(count))
    # subtraces = count
    # print(f"SUBTRACES: {subtraces}")
    # dz_t = complex(m_max[0] - m_min[0] / subtraces, m_max[1] - m_min[1] / subtraces)
    # dc_t = complex(m_max[2] - m_min[2] / subtraces, m_max[3] - m_min[3] / subtraces)
    # dc_t = (m_min[3] - m_min[2]) / subtraces
    # dz_t_chunk = dz_t * workers
    # z0 = complex((m_min[0] + id*dz_t), (m_min[1] + id*dz_t))
    # c0 = complex((m_min[2] + id*dc_t), (m_min[3] + id*dc_t))
    # z0 = complex(m_min[0], m_min[1]) + id*dz_t
    # c0 = complex(m_min[2], m_min[3]) + id*dc_t

    #
    # xtraces = int(count / 1)
    # ytraces =
    # # ytraces = int()
    # current = [rwin.xmin, rwin.ymin]
    # for s in range(count):
    # z0 = 0 + 0j
    progress_increment = (chunks / workers)/100
    chunker = 0
    for chunk in range(id, chunks, workers):
        chunk_col = chunk % chunks_per_side
        chunk_row = math.floor(chunk / chunks_per_side)
        chunk_start = m_min + np.dot(m_chunk_inc, [[chunk_col, 0], [0, chunk_row]])
        # if id == 0:
        #     print(f"CHUNK_POS: {chunk_row}, {chunk_col}")
        #     print(f"CHUNK: {chunk}")
        #     print(f"CHUNK_START: \n{chunk_start}")
        if id == 0:
            progress_milestone(start_time, int((chunker / chunks_per_worker) * 100))
        chunker += 1
        for s0 in range(sqrt_traces_per_chunk):
            for s1 in range(sqrt_traces_per_chunk):
                # chunk_pos = chunk_start + np.dot([s0, s1], m_inc)
                # chunk_pos = chunk_start + np.dot([[s0, s1], [s0, s1]], m_inc)
                chunk_pos = chunk_start + np.dot(m_inc, [[s0, 0], [0, s1]])
                # Only track process from one of the threads, as otherwise it just spams console output
                # z = 0 + 0j
                escapes = False
                # rand0, rand1 = rand_pool[rand_pool_idx], rand_pool[rand_pool_idx+1]
                # rand_pool_idx += 2
                # TODO: This math is definitely wrong, as it was supposed to render the normal buddhabrot if the z/c min/max were set normally
                #       Instead it renders some kind of abyssal horror
                # z = z_min.real + rand0 * z_diff.real + 1j*(z_min.imag+rand0*z_diff.imag)
                # c = c_min.real + rand1 * c_diff.real + 1j*(c_min.imag+rand1*c_diff.imag)
                # z = 0 + 0j
                # NOTE: Also wrong math, but it's interesting
                # z = m_min[0] + rand0 * m_diff[0].real + 1j*(m_min[1] + rand1 * m_diff[1])
                # c = m_min[2] + rand1 * m_diff[2].real + 1j*(m_min[3] + rand0 * m_diff[3])

                # Correct window math, at least more correct than anything else
                # z = m_min[0] + rand0 * m_diff[0].real + 1j*(m_min[1] + rand0 * m_diff[1])
                # c = m_min[2] + rand1 * m_diff[2].real + 1j*(m_min[3] + rand1 * m_diff[3])
                # z = complex(z0.real, z0.imag)
                # c = complex(c0.real, c0.imag)
                m_z = chunk_pos[0, :]
                m_c = chunk_pos[1, :]
                z = complex(m_z[0], m_z[1])
                c = complex(m_c[0], m_c[1])
                # x, y = rwin.plane2xy(c.real, c.imag)
                # if not(x < 0 or x >= xres or y < 0 or y >= yres):
                #     rwin.data[y, x*3] += 1
                # continue
                # m_min[0] + rand0
                # c = ((rwin.xmax - rwin.xmin) * crand[0] + rwin.xmin) + 1j * ((rwin.ymax - rwin.ymin) * crand[1] + rwin.ymin)
                # xpoints = np.empty(max_iter, dtype=float)
                # ypoints = np.empty(max_iter, dtype=float)
                # cdef np.ndarray[np.float_t, ndim=1] xpoints = np.empty(max_iter, dtype=np.float)
                # cdef np.ndarray[np.float_t, ndim=1] ypoints = np.empty(max_iter, dtype=np.float)
                zr, zi = m_z
                cr, ci = m_c
                limit = 0
                for i in range(max_iter):
                    # z = z * z + c
                    zr1 = zr*zr - zi*zi + cr
                    zi = 2*zr*zi + ci
                    zr = zr1
                    if zr*zr + zi*zi > 4:
                        escapes = True
                        break
                    limit += 1
                    xpoints[i] = zr
                    ypoints[i] = zi
                for i in range(limit):
                    coord = rw.plane2xy(xpoints[i], ypoints[i])
                    # Ignore any points outside the render space
                    if coord.x < 0 or coord.x >= xres or coord.y < 0 or coord.y >= yres:
                        continue
                    if escapes and render_outer:
                        rw.data[coord.y*cres.width + coord.x*3] += 1
                        # rw.data[coord.y*cres.width + coord.x*3 + 1] += i
                    # TODO: Leave disabled for now, adds _WAY_ too much render time to do both inner/outer traces
                    #       unless actually intending to use both, even for relatively small trace/iteration counts
                    # elif render_inner:
                    #     for i in range(limit):
                    #         rwin.data[y, x * 3 + 2] += 1  # Non-escaping incrementor
    # TODO: This is fucking awful, there has to be a better way to copy this data over to numpy?
    #       Maybe should just use a cython array to begin with
    cdef array.array data = array.array('i', [])
    array.resize(data, rw.size)
    memcpy(<void*> rw.data, data.data.as_voidptr, rw.size)
    rwin.data = np.asarray(<np.int32_t[:resolution.height, :(resolution.width*3)]> rw.data)
    rwin.serialize(f"render{id}.dat")


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
