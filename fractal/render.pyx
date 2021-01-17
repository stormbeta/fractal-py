#!/usr/bin/env python3

import math
import multiprocessing as mp
from datetime import datetime
import time
from libc.math cimport sqrt

import cython
import numpy as np
cimport numpy as np
import png

from .common import config, Config, progress_milestone, log, frame_params, FrameConfig
from .cmath cimport *
from .iterator cimport p4_iterate
from . import serialization

# FEATURES:
"""
* Somewhat optimized: Use normal mandelbrot/julia render as index for density, and also to skip regions with 100% escaped pixels
* Core loop is near-native Cython/Numpy
* Allow dynamic density of tracing based on coarse iteration count from mandel/julia (minor speed increase, better for increasing contast if desired)
* Deterministic output - Does not rely on random sampling of points, fixed cartesian distribution of variable density used instead
                         This is especially important for rendering animations to avoid flickering/noisy output!
* Parallel processing: Uses Python's multiprocessing library and multiple copies of the pixel buffer to allow parallel computation
* 4D cartesian plane coordinates, allowing for non-standard render planes (no rotation as I don't know how to define that in 4D space)
        TODO: allow use of polar-form coordinates for curved render planes
"""


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

# TODO: This should probably be renamed, it's not really a histogram, just mandelbrot/julia tweaked for optimizing main loop
@cython.cdivision(True)
@cython.infer_types(True)
@cython.boundscheck(False)
cdef np.ndarray[np.uint8_t, ndim=2] render_histogram(RenderConfig histcfg):
    cdef:
        Plane plane = histcfg.rwin.plane
        RenderWindow rwin = histcfg.rwin
        np.ndarray[np.uint8_t, ndim=2] data, data2
        double hist_k
        float theta = frame_params.theta
        int neighbor_sum
        int i = 0
        int x, y
        Point4 point
        double threshold2 = config.escape_threshold*config.escape_threshold
        int skip_optimization = config.skip_hist_optimization

    data = np.full(fill_value=config.min_density, dtype=np.uint8, shape=(rwin.resolution, rwin.resolution))

    with nogil:
        for x in range(rwin.resolution):
            for y in range(rwin.resolution):
                escapes = False
                point = p4_add(histcfg.m_min, p4_dot(histcfg.m_dt, make_p4(x, 0, 0, y)))
                for i in range(histcfg.iteration_limit):
                    point = p4_iterate(point, i, theta)
                    if point.zr * point.zr + point.zi * point.zi > threshold2:
                        data[x, y] = i
                        escapes = True
                        break
                if not (escapes or skip_optimization):
                    data[x, y] = 0
    # Next to the boundary, chunks that "escape" on the histogram actually contain both escaping and non-escaping points
    # And the non-escaping points are always high iteration count, so default to max_density
    # NOTE: You probably want to disable this in some cases where there's no contiguous non-escaping areas
    if not config.skip_hist_boundary_check:
        data2 = np.copy(data)
        with nogil:
            for x in range(1, rwin.resolution - 1):
                for y in range(1, rwin.resolution - 1):
                    neighbor_sum = data[x, y] + data[x + 1, y] + data[x, y + 1] + data[x - 1, y] + data[x, y - 1]
                    if neighbor_sum > 0:
                        data2[x,y] = neighbor_sum / 4
        data = data2
    # Linearly scale histogram to fit min/max density
    # TODO: This code isn't as consistent as it should be for animations
    if config.min_density != config.max_density:
        diff = config.max_density - config.min_density
        data = (data*(config.max_density / np.max(data))).astype(np.uint8)
        return data.clip(1, config.max_density)
    else:
        return data.clip(1, config.max_density)


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.overflowcheck(False)
@cython.infer_types(True)    # NOTE: Huge performance boost
@cython.cdivision(True)      # NOTE: Huge performance boost
def nebula(id: int, shared_data: mp.Array, workers: int, cfg: Config):
    # Allow overrides from main.py
    config.inline_copy(cfg)
    cdef:
        Plane plane = c_plane(config.render_plane)
        np.ndarray[np.uint8_t, ndim=2] histdata
    rwin = RenderWindow(plane, config.global_resolution)

    # IMPORTANT: histogram and main render *must* use the same plane, m_min, and m_max!
    """
    These define the lower and upper points that will be iterated across to form the starting values of Z and C for each trace
    Or in other words, these define the plane representing the cross-section of the 4D space we wish to render
    
    The increment is simplistic - just increment linearly along each axis independently
    Only the values of Z at each iteration are used to determine X,Y pixel coordinates 
    NOTE: No attempt to is made to ensure the plane and render window have matching aspect ratios - I wouldn't even know how to calculate that if I wanted to
    """
    # TODO: This only allows planes that are roughly orthogonal to the cartesian axes
    #       I don't know enough about how to transform points to a plane in 4D space to define other orientations of a plane
    # TODO: Allow using polar-form coordinates for curved 2D surfaces and not just cartesian flat planes
    m_min_list, m_max_list = config.template_m_plane(frame_params.theta)
    m_min, m_max = c_point(m_min_list), c_point(m_max_list)
    rconfig = RenderConfig(rwin, config.iteration_limit, m_min, m_max)

    # TODO: avoid regenerating histogram for every worker? Though honestly unless I parallelize it doesn't really matter
    #       and it's fast enough anyways other than per-frame, and per-frame I probably only want one worker per frame anyways

    # Minimum traces per side of any given chunk (traces per chunk equals density^2)
    # Any chunks containing _only_ escaped points will be skipped entirely
    # Recommended values: 32x32 flat for standard render
    #                      4x32 higher contrast, ~20% faster in some cases
    #                     64x64 for ultra high resolution, or 4x64 for higher contrast
    # NOTE: This isn't really much of a performance optimization, it has a bigger effect on color/contrast
    cdef int min_density = config.min_density
    cdef int max_density = config.max_density

    # Render histogram of mandelbrot set, and linearly scale density to within a configurable min/max
    histwin = RenderWindow(plane, rwin.resolution)
    histcfg = RenderConfig(histwin, pow(2, 7), m_min, m_max)
    histdata = render_histogram(histcfg)
    if id == 0 and config.progress_indicator:
        log.info(f"Render interval: {m_min} => {m_max}")
        log.info(f"log2(traces) = {math.log2(np.sum(np.power(histdata, 2))):.2f}")
        log.info(f"density interval: ({min_density}, {max_density})")
        if config.save_histogram_png:
            serialization.save_histogram_png(histdata)

    rdata = np.full(fill_value=0, dtype=np.float32, shape=config.rshape())
    render2(id, rconfig, histcfg, histdata, rdata, workers)
    # Technically this should probably be in calling code, but this avoids having to manage returning references to temporary data
    with shared_data.get_lock():
        shared = np.frombuffer(shared_data.get_obj(), dtype=np.float32)
        shared.shape = config.rshape()
        shared += rdata


# TODO: Autoscale density/trace count with resolution
#       Likewise, we need to adjust color scaling too due to increased impact of outlier points
# TODO: Separate render plane from pixel plane - points from outside view plane may matter to image
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.overflowcheck(False)
@cython.infer_types(True)    # NOTE: Big performance boost
@cython.cdivision(True)      # NOTE: Big performance boost
def render2(id: int,
            RenderConfig rconfig,
            RenderConfig histcfg,
            np.ndarray[np.uint8_t, ndim=2] histogram,
            np.ndarray[np.float32_t, ndim=3] data,
            workers: int):

    # NOTE: iteration_limit _must_ be less than or equal to 4096 because of the static xpoints/ypoints arrays
    #       it provides a noticeable speedup over dynamic allocation, and animated renders shouldn't have high iteration counts anyways
    # Also, in most cases setting higher iteration limits hasn't proven terribly interesting so far
    # The exception I thought I found I now suspect to actually be precision limits
    assert rconfig.iteration_limit <= 4096
    log.debug(f"Theta: {frame_params.theta}")

    cdef:
        # const double density_factor = 0.5
        # Plane plane = rconfig.rwin.plane
        Plane plane = c_plane(config.view_plane)
        RenderWindow rwin = rconfig.rwin
        int sqrt_chunks = histcfg.rwin.resolution
        double zr_points[4096]
        double zi_points[4096]
        double cr_points[4096]
        double ci_points[4096]
        float theta = frame_params.theta

        # Loop vars
        int i, points, chunk_density, chunk_col, chunk_row
        Point4 p, chunk_start, chunk_pos, chunk_end, chunk_dt
        int chunks = sqrt_chunks*sqrt_chunks
        int chunk_count = 0
        np.ndarray[np.uint32_t, ndim=1] chunk_list
        int count = 0
        double radius
        double threshold2 = config.escape_threshold*config.escape_threshold

    start_time = time.time()

    chunk_list = np.fromiter(range(id, chunks, workers), dtype=np.uint32)
    if id == 0 and config.progress_indicator:
        # Shuffle progress indicator list on process 0 to provide more accurate ETA times
        # If we progress linearly, different regions of the image have different iteration counts per trace
        np.random.shuffle(chunk_list)
        progress_total = np.sum(np.reshape(np.copy(histogram), newshape=(pow(histcfg.rwin.resolution, 2),) )[id::workers])
        log.info(f"log2(chunks): {math.log2(chunks):.2f}")

    # Core rendering loops
    for chunk in chunk_list:
        chunk_col = chunk % sqrt_chunks
        chunk_row = math.floor(chunk / sqrt_chunks)
        chunk_density = histogram[chunk_col, chunk_row]
        if chunk_density == 0:
            log.error("ERROR: Chunk density 0 is impossible")
            continue
        if id == 0 and config.progress_indicator:
            count += chunk_density
            if chunk % 2048 == 0:
                # Only update progress every 128 traces (average)
                progress_milestone(start_time, ((count / progress_total) * 100))
            chunk_count += 1
        chunk_start = p4_add(rconfig.m_min,
                             p4_dot(histcfg.m_dt, make_p4(chunk_col, 0, 0, chunk_row)))
        chunk_dt = p4_scalar_div(histcfg.m_dt, chunk_density)
        with cython.nogil:
            for s0 in range(chunk_density):
                for s1 in range(chunk_density):
                    p = p4_add(chunk_start, p4_dot(chunk_dt, make_p4(s0, 0, 0, s1)))
                    radius = sqrt(p.zr * p.zr + p.zi * p.zi)
                    # radius = math.atan(p.zi/p.zr)
                    escapes = False
                    points = 0
                    for i in range(rconfig.iteration_limit):
                        # p = p4_iterate(p, (i+1)/frame_params.theta)
                        p = p4_iterate(p, i, theta)
                        if p.zr * p.zr + p.zi * p.zi > threshold2:
                            escapes = True
                            break
                        points += 1
                        zr_points[i] = p.zr
                        zi_points[i] = p.zi
                        # cr_points[i] = p.cr
                        # ci_points[i] = p.ci
                    if escapes:
                        for i in range(points):
                            x, y = zr_points[i], zi_points[i]
                            # x, y = cr_points[i], ci_points[i]
                            if plane.xmin < x < plane.xmax and plane.ymin < y < plane.ymax:
                                a, b = rwin.x2column(x), rwin.y2row(y)
                                data[a, b, 0] += 1
                                data[a, b, 1] += i
                                data[a, b, 2] += radius
                            # x, y = cr_points[i], ci_points[i]
                            # if plane.xmin < x < plane.xmax and plane.ymin < y < plane.ymax:
                            #     a, b = rwin.x2column(x), rwin.y2row(y)
                            #     data[a, b, 0] += 1
                            #     data[a, b, 1] += i
