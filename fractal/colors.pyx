import numpy as np
import cython
cimport numpy as np
import math
from .common import config

# input_channel:
#   0: sum(traces)    - how many traces hit this pixel
#   1: sum(iteration) - sum of the iteration count of each trace as it hit this pixel
#   2: sum(z0 radius) - sum of the radius from origin of initial point of trace
# output_channel:
#   0: red, 1: green, 2: blue

@cython.infer_types(True)
@cython.cdivision(True)
def colorize_simple(data: np.ndarray[np.float32]) -> np.ndarray[np.float32]:
    output = np.zeros(dtype=np.uint32, shape=config.rshape())
    # TODO: add colorspace and gradient functions to allow a wider range of coloring schemes
    # NOTE: Keep nmax/imax as constants when rendering animations
    #       Otherwise average brightness could change every frame!
    data = clip_scale(data, 5, 1, 3)
    sqrt_curve(data, output, 1, 0)
    sqrt_curve(data, output, 0, 1)
    sqrt_curve(data, output, 2, 2)
    return output

@cython.infer_types(True)
@cython.cdivision(True)
def colorize_std(data: np.ndarray[np.float32]) -> np.ndarray[np.uint8]:
    output = np.zeros(dtype=np.uint32, shape=config.rshape())
    data[...] = stddev_scale(data, 255, 10.0, 14.0, 20.0)
    # Color channel mapping
    output[:, :, 0] = data[:, :, 0]
    output[:, :, 1] = data[:, :, 1]
    output[:, :, 2] = data[:, :, 2]
    return data

@cython.infer_types(True)
@cython.cdivision(True)
def colorize_percentile(data: np.ndarray[np.float32]) -> np.ndarray[np.uint8]:
    output = np.zeros(dtype=np.uint32, shape=config.rshape())
    data[...] = percentile_scale(data, 255, 0.1, 0.1, 0.1)
    # Color channel mapping
    output[:, :, 0] = data[:, :, 0]
    output[:, :, 1] = data[:, :, 1]
    output[:, :, 2] = data[:, :, 2]
    return output


@cython.infer_types(True)
cdef linear_all(arr, max_value):
    arr_n = arr[:, :, 0]
    arr_i = arr[:, :, 1]
    arr_r = arr[:, :, 2]
    arr_n *= max_value/np.max(arr_n)
    arr_i *= max_value/np.max(arr_i)
    arr_r *= max_value/np.max(arr_r)
    return arr


@cython.infer_types(True)
cdef percentile_scale(arr, max_value, qn, qi, qr):
    arr_n = arr[:, :, 0]
    arr_i = arr[:, :, 1]
    arr_r = arr[:, :, 2]
    arr_n *= max_value/(np.percentile(arr_n, 99 + (1-qn)))
    arr_i *= max_value/(np.percentile(arr_i, 99 + (1-qi)))
    arr_r *= max_value/(np.percentile(arr_r, 99 + (1-qr)))
    return arr


@cython.infer_types(True)
cdef stddev_scale(arr, max_value, nm, im, rm):
    arr_n = arr[:, :, 0]
    arr_i = arr[:, :, 1]
    arr_r = arr[:, :, 2]
    std_n = np.std(arr_n)
    std_i = np.std(arr_i)
    std_r = np.std(arr_r)
    print(f"std: {std_n}, {std_i}, {std_r}")
    avg_n = np.mean(arr_n)
    avg_i = np.mean(arr_i)
    avg_r = np.mean(arr_r)
    arr_n *= max_value/(avg_n + nm*std_n)
    arr_i *= max_value/(avg_i + im*std_i)
    arr_r *= max_value/(avg_r + rm*std_r)
    return np.minimum(max_value, arr)


@cython.infer_types(True)
cdef clip_scale(arr, float nx, float ix, float rx):
    arr_n = arr[:, :, 0]
    arr_i = arr[:, :, 1]
    arr_r = arr[:, :, 2]
    arr[:, :, 0] = np.minimum(arr_n, np.max(arr_n) / nx)
    arr[:, :, 1] = np.minimum(arr_i, np.max(arr_i) / ix)
    arr[:, :, 2] = np.minimum(arr_r, np.max(arr_r) / rx)
    return arr


@cython.infer_types(True)
cdef normalize(arr, int amax):
    amax = 256
    histogram, bins = np.histogram(arr.flatten(), amax, [0, amax])
    # Credit goes to https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_histograms/py_histogram_equalization/py_histogram_equalization.html
    # print(histogram)
    cdf = histogram.cumsum()
    # print(cdf)
    cdf_norm = cdf * histogram.max() / cdf.max()
    cdf_m = np.ma.masked_equal(cdf, 0)
    cdf_m = (cdf_m - cdf_m.min()) * 255 / (cdf_m.max() - cdf_m.min())
    # print(np.max(arr))
    cdf = np.ma.filled(cdf_m, 0).astype(np.uint32)
    return cdf[arr]


@cython.infer_types(True)
cdef log_curve(arr0, arr1, inset, outset, maximum: int):
    k = 255 / math.log2(maximum)
    arr1[:, :, outset] = np.multiply(k, np.log2(arr0[:, :, inset]))

@cython.infer_types(True)
cdef sqrt_curve(arr0, arr1, inset, outset):
    k = 255 / math.sqrt(np.max(arr0[:, :, inset]))
    arr1[:, :, outset] = k * np.sqrt(arr0[:, :, inset])

# @cython.infer_types(True)
# cdef inv_sqrt_curve(arr0, arr1, inset, outset, maximum: int):
#     k = 255 / math.sqrt(maximum)
#     arr1[:, :, outset] = maximum - k * np.sqrt(arr0[:, :, inset])

# @cython.infer_types(True)
# cdef quasi_curve(arr0, arr1, inset, outset, maximum: int):
#     linear_k = 255 / maximum
#     sqrt_k = 255 / math.sqrt(maximum)
#     arr1[:, :, outset] = (sqrt_k * np.sqrt(arr0[:, :, inset]) + linear_k * arr0[:, :, inset]) / 2

@cython.infer_types(True)
cdef linear(arr0, arr1, inset, outset, maximum: int):
    arr1[:, :, outset] = (255 / maximum) * arr0[:, :, inset]

# cdef np.ndarray[np.float32_t, ndim=1] rgb2hsv(np.ndarray[np.float32_t, ndim=1] rgb):
#     cdef:
#         float red = rgb[0]
#         float green = rgb[1]
#         float blue = rgb[2]
#         float cmax = max(red, green, blue)
#         float cmin = min(red, green, blue)
#         float delta = cmax - cmin
#         float hue, saturation, value
#     if delta == 0:
#         hue = 0
#     elif cmax == red:
#         hue = 60 * (((green - blue)/delta) % 6)
#     elif cmax == green:
#         hue = 60 * ((blue - red)/delta + 2)
#     else: # cmax == blue
#         hue = 60 * ((red - green)/delta + 4)
#     hue /= 360
#     if cmax == 0:
#         saturation = 0
#     else:
#         saturation = delta / cmax
#     value = cmax
#     return np.fromiter([hue, saturation, value], dtype=np.float32)
#
#
# # TODO: Need a better way to scale values individually
# cdef np.ndarray[np.float32_t, ndim=1] hsv2rgb(np.ndarray[np.float32_t, ndim=1] hsv):
#     cdef float hue = hsv[0]*360
#     cdef float saturation = hsv[1]
#     cdef float value = hsv[2]
#     cdef float c = value * saturation
#     cdef float x = c * (1 - abs(((hue/60) % 2) - 1))
#     cdef float m = value - c
#     # print(f"c: {c}, x: {x}, m: {m}")
#     cdef float red, green, blue
#     if hue < 60.0:
#         red, green, blue = c, x, 0
#     elif hue < 120.0:
#         red, green, blue = x, c, 0
#     elif hue < 180.0:
#         red, green, blue = 0, x, c
#     elif hue < 240.0:
#         red, green, blue = x, 0, c
#     else:
#         red, green, blue = c, 0, x
#     return np.fromiter([min(255, 255*(red + m)),
#                         min(255, 255*(green + m)),
#                         min(255, 255*(blue + m))],
#                        dtype=np.float32)
