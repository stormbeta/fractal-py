#!/usr/bin/env python3

import math
import multiprocessing
import os
from datetime import datetime
from multiprocessing import Process

from common import *
from rwindow import *

# max_iter = 1000
max_iter = int(math.pow(10, 3.5))
traces = pow(2, 25)
# render_inner = True
render_inner = False
render_outer = True
res = Resolution(1024, 1024)
# res = Resolution(2048, 2048)
plane = Window(-1.75, -1.25, 0.75, 1.25)
# plane = Window(-2.0, -2.0, 2.0, 2.0)
# TODO: CLI interface / args / flags
skip_render = True
# skip_render = False


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


# TODO: combine data in memory maybe? Though via disk has it's own advantages. Maybe generate RAM disk on the fly
def render(id: int, count: int):
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
    # TODO: Random pool is just the default way to handle nebula render
    #       At higher counts, it would be better to actively increment points on the plane
    #       The problem of course is unequal render times if handled naively
    #       I think what I did originally was polar-coordinate staggered slices, but in hindsight
    #       I don't think there's anything wrong with staggered cartesian slices
    rand_pool = np.random.random_sample(count*2+1)
    rand_pool_idx = 0
    for s in range(count):
        # Only track process from one of the threads, as otherwise it just spams console output
        if(id == 0 and s % progress_increment == 0):
            progress_milestone(start_time, int(s / progress_increment))
        # z = 0 + 0j
        escapes = False
        rand0, rand1 = rand_pool[rand_pool_idx], rand_pool[rand_pool_idx+1]
        rand_pool_idx += 2
        # TODO: This math is definitely wrong, as it was supposed to render the normal buddhabrot if the z/c min/max were set normally
        #       Instead it renders some kind of abyssal horror
        # z = z_min.real + rand0 * z_diff.real + 1j*(z_min.imag+rand0*z_diff.imag)
        # c = c_min.real + rand1 * c_diff.real + 1j*(c_min.imag+rand1*c_diff.imag)
        # z = 0 + 0j
        # NOTE: Also wrong math, but it's interesting
        # z = m_min[0] + rand0 * m_diff[0].real + 1j*(m_min[1] + rand1 * m_diff[1])
        # c = m_min[2] + rand1 * m_diff[2].real + 1j*(m_min[3] + rand0 * m_diff[3])

        # Correct window math, at least more correct than anything else
        z = m_min[0] + rand0 * m_diff[0].real + 1j*(m_min[1] + rand0 * m_diff[1])
        c = m_min[2] + rand1 * m_diff[2].real + 1j*(m_min[3] + rand1 * m_diff[3])

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
            if escapes and render_outer:
                rwin.data[y, x*3] += 1      # Baseline incrementor
                rwin.data[y, x*3 + 1] += i  # Based on iteration count to reach this point
            # TODO: Leave disabled for now, adds _WAY_ too much render time to do both inner/outer traces
            #       unless actually intending to use both, even for relatively small trace/iteration counts
            elif render_inner:
                for i in range(limit):
                    rwin.data[y, x * 3 + 2] += 1  # Non-escaping incrementor
    rwin.serialize(f"render{id}.dat")


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
    if skip_render:
        # Allow skipping render to play around with coloring/gradients without re-rendering every single time
        # NOTE: Safe to use on render data from different plane/resolution, as that's stored alongside point data
        rwin = RWindow.deserialize(f"render.dat")
        output = RWindow(resolution=rwin.res, window=rwin.win)
    else:
        start_time = time.time()
        processes = []
        traces_per_process = int(traces / workers)
        print(f"Traces per process: {traces_per_process}")
        for i in range(workers):
            proc = Process(target=render, args=(i, int(traces / workers)))
            processes.append(proc)
            proc.start()
        for proc in processes:
            proc.join()
        print(f"\nElapsed: {seconds_convert(time.time() - start_time)}")

        # MERGE
        rwin = RWindow(resolution=res, window=plane)
        for i in range(workers):
            pdat = RWindow.deserialize(f"render{i}.dat")
            rwin.data += pdat.data
        rwin.serialize("render.dat")
        print(f"Merged {workers} datasets.")
        [os.remove(f"render{i}.dat") for i in range(workers)]
        output = RWindow(resolution=res, window=plane)


    # TODO Convert this to array so that we can combine max + channel offset?
    # Native numpy is ludicrously faster than iterating manually in python
    nmax = np.max(rwin.data[:, 0::3])
    itermax = np.max(rwin.data[:, 1::3])
    innermax = np.max(rwin.data[:, 2::3])
    print(f"Colorizing: [nmax: {nmax}, imax: {itermax}, alt: {innermax}]")

    #         red = (linear_curve(red, rmax) + sqrt_curve(red, rmax)) / 2
    # rwin.data[:, 0::3] = (255/rmax) * rwin.data[:, 0::3]
    # np_sqrt_curve(rwin.data, output.data, 0, 1, nmax/3)
    np_linear(rwin.data, output.data, 0, 1, nmax/3)
    # np_log_curve(rwin.data, 0, rmax*3)
    np_sqrt_curve(rwin.data, output.data, 1, 2, itermax*2)
    np_log_curve(rwin.data, output.data, 0, 0, nmax*1.5)

    # np_linear(rwin.data, 0, rmax/7)
    # np_linear(rwin.data, 1, gmax/7)
    # np_log_curve(rwin.data, 2, bmax*3)

    # Clamp values
    output.data = np.minimum(output.data, 255)

    output_filename = f"renders/nebula_{int(datetime.now().timestamp())}.png"
    # TODO: Include resolution/iteration count in filename?
    write_png(output.data, output.res, output_filename)
    print(f"Saved as: {output_filename}")
