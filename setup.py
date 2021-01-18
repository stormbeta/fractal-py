from setuptools import setup
from Cython.Build import cythonize
import numpy

setup(
    ext_modules=cythonize(["fractal/render.pyx", "fractal/cmath.pyx", "fractal/iterator.pyx"],
                          annotate=True,
                          compiler_directives={'language_level': "3",
                                               'infer_types': True,         # IMPORTANT
                                               'cdivision': True,           # IMPORTANT
                                               'overflowcheck': False,
                                               'wraparound': False,
                                               'boundscheck': False,        # Minimal impact surprisingly
                                               'annotation_typing': True,
                                               'c_api_binop_methods': True  # Doesn't seem to affect runtime
                                               }),
    include_dirs=[numpy.get_include()]
)