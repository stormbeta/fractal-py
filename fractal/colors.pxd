cimport numpy as np

ctypedef np.uint8_t BYTE

cdef:
    struct Color:
        np.uint8_t red, green, blue

    # cdef Color hsv2rgb(np.float32_t hue, np.float32_t saturation, np.float32_t value)