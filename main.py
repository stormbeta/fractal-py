import numpy as np
import pyximport
pyximport.install(language_level=3,
                  setup_args={'include_dirs': np.get_include()})
from fractal.render import *

# max_iter = 1000
# NOTE: max_iter cannot be larger than the fixed x/y points array in render.pyx!
max_iter = int(math.pow(2, 15))
# NOTE: Should always be an even power of two to ensure clean sqrt-ability
traces = pow(2, 26)
# TODO: Remove ability to set non-square resolutions, or else figure out the ugly math instead of cheating with sqrt
# res = Resolution(1024, 1024)
res = Resolution(2048, 2048)
plane = Window(-1.75, -1.25, 0.75, 1.25)
# plane = Window(-2.0, -2.0, 2.0, 2.0)
# TODO: CLI interface / args / flags
# skip_render = True
skip_render = False


if __name__ == '__main__':
    workers = multiprocessing.cpu_count() - 2
    # workers = 10
    if skip_render:
        # Allow skipping render to play around with coloring/gradients without re-rendering every single time
        # NOTE: Safe to use on render data from different plane/resolution, as that's stored alongside point data
        rwin = RWindow.deserialize(f"render.dat")
        output = RWindow(resolution=rwin.res, window=rwin.win)
    else:
        start_time = time.time()
        processes = []
        traces_per_process = int(traces / workers)
        print(f"Traces: {traces}")
        for i in range(workers):
            proc = Process(target=render, args=(i,
                                                res,
                                                plane,
                                                max_iter,
                                                workers,
                                                traces))
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
    imax = np.max(rwin.data[:, 1::3])
    innermax = np.max(rwin.data[:, 2::3])
    print(f"Colorizing: [nmax: {nmax}, imax: {imax}, alt: {innermax}]")

    #         red = (linear_curve(red, rmax) + sqrt_curve(red, rmax)) / 2
    # rwin.data[:, 0::3] = (255/rmax) * rwin.data[:, 0::3]
    # np_sqrt_curve(rwin.data, output.data, 0, 0, nmax*3)
    np_linear(rwin.data, output.data, 0, 2, nmax/3)
    # np_sqrt_curve(rwin.data, output.data, 0, 1, nmax)
    np_sqrt_curve(rwin.data, output.data, 1, 1, imax/2)
    """
    np_log_curve(rwin.data, output.data, 0, 0, nmax*1.5)
    np_sqrt_curve(rwin.data, output.data, 0, 1, nmax/3)
    # np_log_curve(rwin.data, 0, rmax*3)
    np_sqrt_curve(rwin.data, output.data, 1, 2, itermax*1.5)
    # np_inv_sqrt_curve(rwin.data, output.data, 1, 2, itermax*1.5)
    """

    # np_linear(rwin.data, 0, rmax/7)
    # np_linear(rwin.data, 1, gmax/7)
    # np_log_curve(rwin.data, 2, bmax*3)

    # Clamp values
    output.data = np.minimum(output.data, 255)

    output_filename = f"renders/nebula_{int(datetime.now().timestamp())}.png"
    # TODO: Include resolution/iteration count in filename?
    write_png(output.data, output.res, output_filename)
    print(f"Saved as: {output_filename}")
