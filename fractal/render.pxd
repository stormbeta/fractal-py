cimport numpy as np
from .data cimport RenderConfig

cdef np.ndarray[np.uint8_t, ndim=2] render_histogram(RenderConfig rconfig, int min_density, int max_density)