# fractal-py

Random nebulabrot fractal nonsense written in python/cython/numpy

## Setup

```bash
# Required
pip install numpy==1.19.3
pip install cython
pip install qtoml

# Prefer lycon, but can use pypng if you can't install lycon for some reason
pip install lycon
pip install pypng

# If you want the coloring GUI to work
pip install pyqt5

python main.py
```

You'll need a working C compiler for Cython to work, e.g. gcc
On Windows you can use the official python build, but it's probably easier to use WSL

No CLI yet, most config is driven by the `config.toml` file.

### Performance Notes

For the iteration function, avoid direct struct initialization, as it has _severe_ impact on performance for some reason - possibly related to this issue: https://github.com/cython/cython/issues/1642

Likewise, use `nogil` to ensure inner loop functions are as close to pure C as possible - this can make up to 3x difference in performance for something as simple as accidentally using a Python type conversion instead of C conversion