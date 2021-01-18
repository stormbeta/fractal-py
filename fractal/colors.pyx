import numpy as np
import cython
cimport numpy as np
import math
from .common import config, log
from typing import Any, Callable, List
from nptyping import NDArray


# colorScale = [red: [n, i, r], green: [n, i, r], blue: [n, i, r]
#   n: sum(traces)    - how many traces hit this pixel
#   i: sum(iteration) - sum of the iteration count of each trace as it hit this pixel
#   r: sum(z0 radius) - not functioning correctly atm


RDATA = NDArray[(Any, Any, 3), np.float32]
COLOR1F = NDArray[(Any, Any, 3), np.float32]
COLOR8F = NDArray[(Any, Any, 3), np.float32]
COLOR8U = NDArray[(Any, Any, 3), np.uint8]
ScalerFunc = Callable[[NDArray[(Any, Any, 3), np.float32], List[float]], NDArray[(Any, Any), np.float32]]

# TODO: This file needs a lot of cleanup after all the changes that were made

# TODO:
# These are a good start, but they still aren't exactly chainable
# TODO: Ability to define actual color gradients? Not sure how that would work precisely
# TODO: Alternatives to clipping, as that washes out bright outliers if they're more than a few pixels large
#       Direct wrapping is bad as it creates a hard min/max value cutoff, maybe support wave transform?

# NOTE: If doing animated work, you MUST use percentile-based schemes or linear
#       Anything else will cause flickering due to uneven brightness between frames
#       Honstly the percentile-based scaling is better anyways


@cython.infer_types(True)
@cython.cdivision(True)
def sqrt_curve(rdata: np.ndarray[np.float32], scale: List[float]):
    output_f = np.zeros(dtype=np.float32, shape=(config.global_resolution, config.global_resolution))
    for i in range(3):
        if scale[i] > 0.0:
            # Fix value cap for linear
            m = np.max(rdata[:, :, i])
            # TODO: This should be the caller's responsibility to make it a smaller increment?
            divisor = scale[i]/10
            if m / divisor <= 0.0:
                k = 1.0
            else:
                k = 1.0 / math.sqrt(m / divisor)
            output_f += k * np.sqrt(np.clip(rdata[:, :, i], 0, m / divisor))
    out_max = np.max(output_f)
    if out_max > 0.0:
        output_f *= (1.0 / out_max)
    return output_f


@cython.infer_types(True)
@cython.cdivision(True)
def half_sine(rdata: np.ndarray[np.float32], scale: List[float]):
    output_f = np.zeros(dtype=np.float32, shape=(config.global_resolution, config.global_resolution))
    for i in range(3):
        if scale[i] > 0.0:
            m = np.max(rdata[:, :, i])
            k = math.pi / (2 * m)
            output_f += np.sin(k * np.clip(rdata[:, :, i], 0, m/(scale[i]/10)))
    out_max = np.max(output_f)
    if out_max > 0.0:
        output_f *= (1.0 / out_max)
    return output_f


@cython.infer_types(True)
@cython.cdivision(True)
def percentile(rdata: np.ndarray[np.float32], scale: List[float]):
    output_f = np.zeros(dtype=np.float32, shape=(config.global_resolution, config.global_resolution))
    for i in range(3):
        if scale[i] > 0.0:
            q = np.percentile(rdata[:, :, i], 99 + (1 - (scale[i]/config.color_scale_max)))
            if q == 0:
                continue
            k = 1 / q
            m = np.max(rdata[:, :, i])
            k = 1.0 / math.sqrt(q)
            output_f += k * np.sqrt(np.clip(rdata[:, :, i], 0, q))
    out_max = np.max(output_f)
    if out_max > 0.0:
        output_f *= (1.0 / out_max)
    return output_f


@cython.infer_types(True)
@cython.cdivision(True)
def percentile_sine(rdata: np.ndarray[np.float32], scale: List[float]):
    output_f = np.zeros(dtype=np.float32, shape=(config.global_resolution, config.global_resolution))
    for i in range(3):
        if scale[i] > 0.0:
            q = np.percentile(rdata[:, :, i], 98.0 + (2.0 - 2.0*(scale[i]/config.color_scale_max)))
            k = 1 / q
            m = np.max(rdata[:, :, i])
            k = math.pi / (2 * q)
            output_f += np.sin(k * np.clip(rdata[:, :, i], 0, q))
    out_max = np.max(output_f)
    if out_max > 0.0:
        output_f *= (1.0 / out_max)
    return output_f


@cython.infer_types(True)
@cython.cdivision(True)
def ln_curve(rdata: np.ndarray[np.float32], scale: List[float]):
    output_f = np.zeros(dtype=np.float32, shape=(config.global_resolution, config.global_resolution))
    for i in range(3):
        if scale[i] > 1.0:
            m = np.max(rdata[:, :, i])
            adj = scale[i]/10 - 1
            k = 1.0 / math.log(m)
            output_f += k * np.log((rdata[:, :, i] / adj) + 1)
    out_max = np.max(output_f)
    if out_max > 0.0:
        output_f *= (1.0 / out_max)
    return output_f


@cython.infer_types(True)
@cython.cdivision(True)
# TODO: Probably get rid of this one? Not very useful
def full_sine(rdata: np.ndarray[np.float32], scale: List[float]):
    output_f = np.zeros(dtype=np.float32, shape=(config.global_resolution, config.global_resolution))
    for i in range(3):
        if scale[i] > 0.0:
            m = np.max(rdata[:, :, i])
            m_adj = m/(scale[i]/10)
            k = math.pi / m
            output_f += np.sin(k * np.clip(rdata[:, :, i], 0, m_adj) - (math.pi/2)) + 1.0
    out_max = np.max(output_f)
    if out_max > 0.0:
        output_f *= (1.0 / out_max)
    return output_f


@cython.infer_types(True)
@cython.cdivision(True)
def linear(rdata: np.ndarray[np.float32], scale: List[float]):
    output_f = np.zeros(dtype=np.float32, shape=(config.global_resolution, config.global_resolution))
    for i in range(3):
        if scale[i] > 0.0:
            m = np.max(rdata[:, :, i])
            m_adj = m/(scale[i]/10)
            output_f += (1.0 / m_adj) * np.clip(rdata[:, :, i], 0, m_adj)
    out_max = np.max(output_f)
    if out_max > 0.0:
        output_f *= (1.0 / out_max)
    return output_f


@cython.infer_types(True)
@cython.cdivision(True)
def colorize_simple2(rdata: np.ndarray[np.float32],
                     scaling: List[List[float]],
                     scale_funcs: List[ScalerFunc] = None) -> COLOR8U:
    if scale_funcs is None:
        scale_funcs = [sqrt_curve, sqrt_curve, sqrt_curve]
    rgb_f = np.zeros(dtype=np.float32, shape=config.rshape())
    cmax: List[float] = [np.max(rdata[:, :, 0]), np.max(rdata[:, :, 1]), np.max(rdata[:, :, 2])]
    log.info(f"maxes: {cmax}")
    i: int = 0
    for color in scaling:
        rgb_f[:, :, i::3] += scale_funcs[i](rdata, color).reshape((config.global_resolution, config.global_resolution, 1))
        i += 1
    return (255.0 * rgb_f).astype(np.uint8)


scalers = {
    'sqrt': sqrt_curve,
    'half_sine': half_sine,
    'percentile': percentile,
    'percentile_sine': percentile_sine,
    'linear': linear,
    'ln_curve': ln_curve,
}


# TODO: Find an image/color processing library for HSV/RGB conversion, it's a headache to do manually
#       More importantly: expose HSV/HSL as options in GUI
