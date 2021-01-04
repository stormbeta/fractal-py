from setuptools import setup
from Cython.Build import cythonize

setup(
    ext_modules=cythonize("fractal/render.pyx",
                          annotate=True,
                          compiler_directives={'language_level': "3",
                                               'infer_types': True,
                                               'cdivision': True,
                                               'overflowcheck': True,
                                               'wraparound': False,
                                               'boundscheck': False
                                               })
)