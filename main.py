import ctypes
import logging
import os

import numpy as np
import multiprocessing as mp
import time
from datetime import datetime
import pyximport

from fractal.common import seconds_convert, MultiFrameConfig, config, frame_params, log

# NOTE: This must be done before loading pyximport, or else Cython compilation will fail
with open('fractal/iterator.pyx', 'w') as fp:
    fp.write("\n".join([
        "import math",
        "from .cmath cimport Point4",
        config.iteration_func]))

with open('fractal/iterator.pxd', 'w') as fp:
    fp.write("\n".join([
        "from .cmath cimport Point4",
        "cdef Point4 p4_iterate(Point4 a, int i)"
    ]))

pyximport.install(language_level=3,
                  setup_args={'include_dirs': np.get_include()})

from fractal.render import nebula
from fractal.colors import colorize_simple2
from fractal import serialization


# skip_render = True
skip_render = False


def render_frame(theta: float, workers: int, number: int = -1):
    shared_data = mp.Array(ctypes.c_float, pow(config.global_resolution, 2)*3)
    if skip_render:
        data = serialization.load_render_dat()
    else:
        start_time = time.time()
        processes = []
        for i in range(workers):
            proc = mp.Process(target=nebula, args=(i, shared_data, workers, config, theta))
            processes.append(proc)
            proc.start()
        for proc in processes:
            proc.join()
        log.info(f"\nElapsed: {seconds_convert(time.time() - start_time)}")
        data = np.frombuffer(shared_data.get_obj(), dtype=np.float32)
        data.shape = config.rshape()
        if config.save_render_data:
            serialization.save_render_dat(data)
    # output = colorize_simple(data, [12, 2, 8], [2, 0, 1])
    output = colorize_simple2(data, [[0, 0, 0], [0, 8, 0], [0, 8, 0]])
    output[...] = np.minimum(255, output)
    serialization.save_render_png(output, number)


def multirender(id: int, workers: int, params: MultiFrameConfig):
    frame_params.inline_copy(params)
    t_delta: float = (config.stop - config.start) / config.frames
    log.warning("Disabling frame progress indicator, render.dat, and histogram png for multi-frame render")
    config.progress_indicator = False
    config.save_render_data = False
    config.save_histogram_png = False
    for frame in range(id, config.frames, workers):
        theta: float = config.start + t_delta*frame
        frame_params.frame = frame
        frame_params.theta = theta
        render_frame(theta, 1, frame)


if __name__ == '__main__':
    log.info(f"Resolution: {config.global_resolution}")

    # TODO: Move to config.toml
    # start = config.start
    # stop = config.stop
    # # frames = 24*60
    # # start = -2*math.pi
    # frames = config.frames
    # # frames = 1

    workers = config.workers
    log = logging.getLogger(mp.current_process().name)
    if config.frames == 1:
        log.info("Single frame render mode")
        render_frame(config.start, workers, -1)
    else:
        log.info("Multi-frame render mode")
        processes = []
        for i in range(workers):
            frame_params.folder = f"frames/nebula-{int(datetime.now().timestamp())}"
            os.mkdir(frame_params.folder)
            proc = mp.Process(name=f"worker-{i}",
                              target=multirender,
                              args=(i, workers, frame_params))
            processes.append(proc)
            proc.start()
        for proc in processes:
            proc.join()