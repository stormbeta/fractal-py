import numpy as np
import pyximport
import sys
pyximport.install(language_level=3,
                  setup_args={'include_dirs': np.get_include()})
from fractal.render import *
from fractal.common import global_resolution

# max_iter = 1000
# NOTE: max_iter cannot be larger than the fixed x/y points array in render.pyx!
# max_iter = int(math.pow(2, 16))
# max_iter = int(math.pow(2, 10))
# NOTE: Should always be an even power of two to ensure clean sqrt-ability
# traces = pow(2, 22)
# res = Resolution(1024, 1024)
# res = Resolution(2048, 2048)
# plane = Window(-1.75, -1.25, 0.75, 1.25)
# plane = Window(-2.0, -2.0, 2.0, 2.0)
# TODO: CLI interface / args / flags
# skip_render = True
skip_render = False

def rendrend(angle: float):
    # workers = multiprocessing.cpu_count()
    workers = 12
    resolution = global_resolution

    if not skip_render:
        start_time = time.time()
        processes = []
        for i in range(workers):
            proc = Process(target=dothing, args=(i, workers, angle))
            processes.append(proc)
            proc.start()
        for proc in processes:
            proc.join()
        print(f"\nElapsed: {seconds_convert(time.time() - start_time)}")

        data = np.zeros(dtype=np.uint32, shape=(resolution, resolution * 3))
        for i in range(workers):
            with open(f"render{i}.dat", "rb") as fp:
                load = np.frombuffer(fp.read(), dtype=np.uint32)
                load.shape = (resolution, resolution * 3)
                # output_filename = f"renders/wat/wat_{i}.png"
                # with open(output_filename, "wb") as fp:
                #     writer = png.Writer(resolution, resolution, greyscale=False)
                #     writer.write(fp, load.astype('uint8'))
                data += load
                # with open('render.dat', 'wb') as fp:
                #     fp.write(data.tobytes())
    else:
        with open(f"render.dat", "rb") as fp:
            data = np.frombuffer(fp.read(), dtype=np.uint32)
            data.shape = (resolution, resolution * 3)

    output = np.zeros(dtype=np.uint32, shape=(resolution, resolution * 3))
    nmax = np.max(data[:, 0::3])
    print(f"nmax: {nmax}")
    nmax = 320
    # imax = np.max(data[:, 1::3])
    np_sqrt_curve(data, output, 0, 2, nmax)
    np_linear(data, output, 0, 1, nmax*0.60)
    output = np.minimum(255, output)

    output_filename = f"renders/nebula_{int(datetime.now().timestamp())}.png"
    with open(output_filename, "wb") as fp:
        writer = png.Writer(resolution, resolution, greyscale=False)
        writer.write(fp, output.astype('uint8'))


if __name__ == '__main__':

    theta = (math.pi/120) * 40
    theta_end = (math.pi/120) * 60
    while theta < math.pi:
        rendrend(theta)
        theta += math.pi/1200

    # workers = 10
    # if skip_render:
    #     # Allow skipping render to play around with coloring/gradients without re-rendering every single time
    #     # NOTE: Safe to use on render data from different plane/resolution, as that's stored alongside point data
    #     rwin = RWindow.deserialize(f"render.dat")
    #     output = RWindow(resolution=rwin.res, window=rwin.win)
    # else:
    #     start_time = time.time()
    #     processes = []
    #     traces_per_process = int(traces / workers)
    #     print(f"Traces: {traces}")
    #     for i in range(workers):
    #         mandel = np.empty((1024, 1024), dtype=np.uint16)
    #         proc = Process(target=render, args=(i,
    #                                             res,
    #                                             plane,
    #                                             max_iter,
    #                                             workers,
    #                                             traces, mandel))
    #         processes.append(proc)
    #         proc.start()
    #     for proc in processes:
    #         proc.join()
    #     print(f"\nElapsed: {seconds_convert(time.time() - start_time)}")
    #
    #     # MERGE
    #     rwin = RWindow(resolution=res, window=plane)
    #     for i in range(workers):
    #         pdat = RWindow.deserialize(f"render{i}.dat")
    #         rwin.data += pdat.data
    #     rwin.serialize("render.dat")
    #     print(f"Merged {workers} datasets.")
    #     [os.remove(f"render{i}.dat") for i in range(workers)]
    #     output = RWindow(resolution=res, window=plane)
    #
    #
    # # TODO Convert this to array so that we can combine max + channel offset?
    # # Native numpy is ludicrously faster than iterating manually in python
    # nmax = np.max(rwin.data[:, 0::3])
    # imax = np.max(rwin.data[:, 1::3])
    # innermax = np.max(rwin.data[:, 2::3])
    # print(f"Colorizing: [nmax: {nmax}, imax: {imax}, alt: {innermax}]")
    #
    # #         red = (linear_curve(red, rmax) + sqrt_curve(red, rmax)) / 2
    # # rwin.data[:, 0::3] = (255/rmax) * rwin.data[:, 0::3]
    # # np_sqrt_curve(rwin.data, output.data, 0, 0, nmax*3)
    # np_linear(rwin.data, output.data, 0, 2, nmax/3)
    # # np_sqrt_curve(rwin.data, output.data, 0, 1, nmax)
    # np_sqrt_curve(rwin.data, output.data, 1, 1, imax/3)
    # """
    # np_log_curve(rwin.data, output.data, 0, 0, nmax*1.5)
    # np_sqrt_curve(rwin.data, output.data, 0, 1, nmax/3)
    # # np_log_curve(rwin.data, 0, rmax*3)
    # np_sqrt_curve(rwin.data, output.data, 1, 2, itermax*1.5)
    # # np_inv_sqrt_curve(rwin.data, output.data, 1, 2, itermax*1.5)
    # """
    #
    # # np_linear(rwin.data, 0, rmax/7)
    # # np_linear(rwin.data, 1, gmax/7)
    # # np_log_curve(rwin.data, 2, bmax*3)
    #
    # # Clamp values
    # output.data = np.minimum(output.data, 255)
    #
    # output_filename = f"renders/nebula_{int(datetime.now().timestamp())}.png"
    # # TODO: Include resolution/iteration count in filename?
    # write_png(output.data, output.res, output_filename)
    # print(f"Saved as: {output_filename}")
