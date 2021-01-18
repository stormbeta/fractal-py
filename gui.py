import pyximport
import numpy as np

pyximport.install(language_level=3,
                  setup_args={'include_dirs': np.get_include()})

from fractal.gui import run_app


if __name__ == '__main__':
    print("NOTE: This UI is only for colorization, it can't invoke rendering yet")
    run_app()