import numpy as np
import cython
cimport numpy as np
import math
from .common import config
from typing import List, Any
from nptyping import NDArray


# colorScale = [red: [n, i, r], green: [n, i, r], blue: [n, i, r]
#   n: sum(traces)    - how many traces hit this pixel
#   i: sum(iteration) - sum of the iteration count of each trace as it hit this pixel
#   r: sum(z0 radius) - not functioning correctly atm


RDATA = NDArray[(Any, Any, 3), np.float32]
COLOR1F = NDArray[(Any, Any, 3), np.float32]
COLOR8F = NDArray[(Any, Any, 3), np.float32]
COLOR8U = NDArray[(Any, Any, 3), np.uint8]

# TODO: This file needs a lot of cleanup after all the changes that were made

# TODO:
# These are a good start, but they still aren't exactly chainable
# TODO: Ability to define actual color gradients? Not sure how that would work precisely
# TODO: Alternatives to clipping, as that washes out bright outliers if they're more than a few pixels large
#       Direct wrapping is bad as it creates a hard min/max value cutoff, maybe support wave transform?

# TODO: Log metrics of color output, or maybe even a basic histogram bin, so that we have reference points for coloring schemes

# TODO: Find out why I keep seeing such significant differences in average brightness for multi-frame renders
#       Even when setting static nmax/imax... it's almost as if the actual trace values are changing drastically, which doesn't make sense
cdef sqrt_curve(rdata: np.ndarray[np.float32], scale: List[float]):
    output_f = np.zeros(dtype=np.float32, shape=(config.global_resolution, config.global_resolution))
    for i in range(3):
        if scale[i] > 0.0:
            m = np.max(rdata[:, :, i])
            # Fix value cap for linear
            # rdata[:, :, i] *= 50000 / m
            # TODO: This should be the caller's responsibility to make it a smaller increment
            # TODO: This should be a multiplier, not a divisor
            divisor = scale[i]/10
            if m / divisor == 0.0:
                k = 1.0
            else:
                k = 1.0 / math.sqrt(m / divisor)
            output_f += k * np.sqrt(np.clip(rdata[:, :, i], 0, m / divisor))
    output_f *= (1.0 / np.max(output_f))
    return output_f


@cython.infer_types(True)
@cython.cdivision(True)
def colorize_simple2(rdata: np.ndarray[np.float32], scaling: List[List[float]]) -> COLOR8U:
    rgb_f = np.zeros(dtype=np.float32, shape=config.rshape())
    cmax: List[float] = [0, 0, 0]
    i: int = 0
    for color in scaling:
        rgb_f[:, :, i::3] += sqrt_curve(rdata, color).reshape((config.global_resolution, config.global_resolution, 1))
        i += 1
    return (255 * rgb_f).astype(np.uint8)


# TODO: Find an image/color processing library for HSV/RGB conversion, it's a headache to do manually
#       More importantly: expose HSV/HSL as options in GUI
