import ctypes
import logging

import numpy as np
import pyximport

pyximport.install(language_level=3,
                  setup_args={'include_dirs': np.get_include()})

from fractal.render import *
from fractal.common import seconds_convert
from fractal.colors import *
from fractal import manager


# skip_render = True
skip_render = False


def render_frame(theta: float, workers: int, number: int = -1):
    shared_data = mp.Array(ctypes.c_float, pow(config.global_resolution, 2)*3)
    if skip_render:
        data = manager.load_render_dat()
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
            manager.save_render_dat(data)
    output = colorize_simple(data, [6, 0.1, 5], [2, 0, 1])
    output[...] = np.minimum(255, output)
    manager.save(output, number)


def multirender(id: int, workers: int, start: float, stop: float, frames: int):
    t_delta: float = (stop - start) / frames
    log.warning("Disabling frame progress indicator, render.dat, and histogram png for multi-frame render")
    config.progress_indicator = False
    config.save_render_data = False
    config.save_histogram_png = False
    for frame in range(id, frames, workers):
        theta: float = start + t_delta*frame
        render_frame(theta, 1, frame)


if __name__ == '__main__':
    log.info(f"Resolution: {config.global_resolution}")

    # TODO: Move to config.toml
    start  = 0.0
    stop   = math.pi/2
    # frames = 24*60
    # start = -2*math.pi
    # frames = 2400
    frames = 1

    workers = config.workers
    log = logging.getLogger(mp.current_process().name)
    if frames == 1:
        log.info("Single frame render mode")
        render_frame(start, workers, -1)
    else:
        log.info("Multi-frame render mode")
        processes = []
        for i in range(workers):
            proc = mp.Process(name=f"worker-{i}",
                              target=multirender,
                              args=(i, workers, start, stop, frames))
            processes.append(proc)
            proc.start()
        for proc in processes:
            proc.join()