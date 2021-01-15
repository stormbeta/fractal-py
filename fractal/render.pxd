cimport numpy as np
from .cmath cimport RenderConfig

cdef np.ndarray[np.uint8_t, ndim=2] render_histogram(RenderConfig rconfig)