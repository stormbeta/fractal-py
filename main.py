import multiprocessing as mp
import ctypes
from datetime import datetime
import png
import sys

import numpy as np
import pyximport

pyximport.install(language_level=3,
                  setup_args={'include_dirs': np.get_include()})

from fractal.render import *
from fractal.common import config, seconds_convert
from fractal.colors import *


# skip_render = True
skip_render = False


def render_frame(theta: float, workers: int, number: int = -1):
    resolution = config.global_resolution
    shared_data = mp.Array(ctypes.c_float, pow(config.global_resolution, 2)*3)

    if not skip_render:
        start_time = time.time()
        processes = []
        for i in range(workers):
            proc = mp.Process(target=nebula, args=(i, shared_data, workers, config, theta))
            processes.append(proc)
            proc.start()
        for proc in processes:
            proc.join()
        print(f"\nElapsed: {seconds_convert(time.time() - start_time)}")
        data = np.frombuffer(shared_data.get_obj(), dtype=np.float32)
        data.shape = config.rshape()
        if config.save_render_data:
            with open('render.dat', 'wb') as fp:
                fp.write(data.tobytes())
    else:
        with open(f"render.dat", "rb") as fp:
            data = np.frombuffer(fp.read(), dtype=np.float32)
            data.shape = config.rshape()
            data = data.copy()

    output = colorize_percentile(data)
    output[...] = np.minimum(255, output)

    if number == -1:
        output_filename = f"renders/nebula-{int(datetime.now().timestamp())}.png"
    else:
        output_filename = f"frames/nebula-{number:04d}-{int(datetime.now().timestamp())}.png"
    with open(output_filename, "wb") as fp:
        writer = png.Writer(resolution, resolution, greyscale=False)
        writer.write(fp, output.astype('uint8').reshape(resolution, resolution * 3))


def multirender(id: int, workers: int, start: float, stop: float, frames: int):
    t_delta: float = (stop - start) / frames
    print("Disabling frame progress indicator, render.dat, and histogram png for multi-frame render")
    config.progress_indicator = False
    config.save_render_data = False
    config.save_histogram_png = False
    for frame in range(id, frames, workers):
        theta: float = start + t_delta*frame
        render_frame(theta, 1, frame)


if __name__ == '__main__':
    print(f"Resolution: {config.global_resolution}")

    # TODO: Move to config.toml
    start  = 0.0
    stop   = math.pi/2
    # frames = 24*60
    # start = -2*math.pi
    # frames = 2400
    frames = 1

    workers = config.workers
    log = mp.get_logger()
    if frames == 1:
        log.info("Single frame render mode")
        render_frame(start, workers, -1)
    else:
        log.info("Multi-frame render mode")
        processes = []
        for i in range(workers):
            proc = mp.Process(target=multirender, args=(i, workers, start, stop, frames))
            processes.append(proc)
            proc.start()
        for proc in processes:
            proc.join()