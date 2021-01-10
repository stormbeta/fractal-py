import numpy as np
import cython
cimport numpy as np
import math
from .common import config


@cython.infer_types(True)
@cython.cdivision(True)
def colorize(data: np.ndarray[np.float32]) -> np.ndarray[np.float32]:
    output = np.zeros(dtype=np.float32, shape=config.rshape())
    # TODO: Configure coloring as a separate step/function for easier experimentation
    # TODO: add colorspace and gradient functions to allow a wider range of coloring schemes
    # TODO: Convert render data to be floating point instead of integer, only the PNG output needs to be uint8
    # NOTE: Keep nmax/imax as constants when rendering animations
    #       Otherwise average brightness could change every frame!
    nmax = np.max(data[:, :, 0])
    imax = np.max(data[:, :, 1])
    rmax = np.max(data[:, :, 2])
    print(f"nmax: {nmax}, imax: {imax}, rmax: {rmax}")
    # Color function args: (input_data, output_data, input_channel, output_color, max_value)
    # input_channel:
    #   0: sum(traces)    - how many traces hit this pixel
    #   1: sum(iteration) - sum of the iteration count of each trace as it hit this pixel
    #   2: sum(z0 radius) - sum of the radius from origin of initial point of trace
    # output_channel:
    #   0: red, 1: green, 2: blue
    # max_value: this should correspond to nmax for input0, or imax for input1
    #            since there's often outliers, you can inversely scale brightness by adding a constant scaling factor to the max value
    nmax = nmax / 5
    imax = imax / 1
    rmax = rmax / 3
    data[:, :, 0] = np.clip(data[:, :, 0], 0, nmax)
    data[:, :, 1] = np.clip(data[:, :, 1], 0, imax)
    data[:, :, 2] = np.clip(data[:, :, 2], 0, rmax)

    linear(data, output, 1, 2, imax)
    sqrt_curve(data, output, 0, 0, nmax)
    sqrt_curve(data, output, 2, 1, rmax)
    return output


@cython.infer_types(True)
cdef log_curve(arr0, arr1, inset, outset, maximum: int):
    k = 255 / math.log2(maximum)
    arr1[:, :, outset] = np.multiply(k, np.log2(arr0[:, :, inset]))

@cython.infer_types(True)
cdef sqrt_curve(arr0, arr1, inset, outset, maximum: int):
    k = 255 / math.sqrt(maximum)
    arr1[:, :, outset] = k * np.sqrt(arr0[:, :, inset])

@cython.infer_types(True)
cdef inv_sqrt_curve(arr0, arr1, inset, outset, maximum: int):
    k = 255 / math.sqrt(maximum)
    arr1[:, :, outset] = maximum - k * np.sqrt(arr0[:, :, inset])

@cython.infer_types(True)
cdef quasi_curve(arr0, arr1, inset, outset, maximum: int):
    linear_k = 255 / maximum
    sqrt_k = 255 / math.sqrt(maximum)
    arr1[:, :, outset] = (sqrt_k * np.sqrt(arr0[:, :, inset]) + linear_k * arr0[:, :, inset]) / 2

@cython.infer_types(True)
cdef linear(arr0, arr1, inset, outset, maximum: int):
    arr1[:, :, outset] = (255 / maximum) * arr0[:, :, inset]

# TODO: Need a better way to scale values individually
# cdef Color hsv2rgb(np.float32_t hue, np.float32_t saturation, np.float32_t value):
#     cdef float c = value * saturation
#     cdef float x = c * (1 - abs((hue/(math.pi/6) % 2 - 1)))
#     cdef float m = value - c
#     cdef float red, green, blue
#     if hue < math.pi/6:
#         red, green, blue = c, x, 0
#     elif hue < math.pi/3:
#         red, green, blue = x, c, 0
#     elif hue < math.pi:
#         red, green, blue = 0, x, c
#     elif hue < 2*math.pi/3:
#         red, green, blue = x, 0, c
#     else:
#         red, green, blue = c, 0, x
#     return Color(max(255, 255*(red + m)),
#                  max(255, 255*(green + m)),
#                  max(255, 255*(blue + m)))
