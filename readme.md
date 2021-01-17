# fractal-py

Random nebulabrot fractal nonsense written in python/cython/numpy

```bash
pip install numpy==1.19.3
pip install pypng
pip install cython

python main.py
```

You'll need a working C compiler for Cython to work, e.g. gcc
On Windows you can use the official python build, but it's probably easier to use WSL

No CLI yet, all config is still in code as I've constantly been tweaking it

### Performance Notes

For the iteration function, avoid direct struct initialization, as it has _severe_ impact on performance for some reason - possibly related to this issue: https://github.com/cython/cython/issues/1642

Likewise, use `nogil` to ensure inner loop functions are as close to pure C as possible - this can make up to 3x difference in performance for something as simple as accidentally using a Python type conversion instead of C conversion