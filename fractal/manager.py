import pkg_resources
from datetime import datetime
import zipfile

from .colors import *
from .common import config

fast_png: bool = 'lycon' in {pkg.key for pkg in pkg_resources.working_set}

if fast_png:
    import lycon
else:
    import png


# TODO: This file should honestly be renamed "data", and the data.pyx file renamed to something like "cmath"


def save(output, number: int = -1):
    resolution = config.global_resolution
    if number == -1:
        output_filename = f"renders/nebula-{int(datetime.now().timestamp())}.png"
    else:
        output_filename = f"frames/nebula-{number:04d}-{int(datetime.now().timestamp())}.png"
    if fast_png:
        # Lycon is considerably faster than pypng, especially at larger resolutions where pypng is ridiculously slow
        # But it also kind of assumes GNU toolchain, and has dependencies on libpng-dev + libjpeg-dev
        lycon.save(output_filename, output.astype('uint8').reshape(resolution, resolution, 3))
    else:
        with open(output_filename, "wb") as fp:
            writer = png.Writer(resolution, resolution, greyscale=False)
            writer.write(fp, output.astype('uint8').reshape(resolution, resolution * 3))


def load_render_dat(file: str = 'render.zip') -> np.ndarray:
    with zipfile.ZipFile(file, 'r') as zp:
        # Mutating global state as a side effect like this is kind of hacky, but config _is_ meant to be global
        config.reload(zp.open('config.toml').read())
        data = np.frombuffer(zp.open('render.dat').read(), dtype=np.float32)
        data.shape = config.rshape()
        return data.copy()


def save_render_dat(data: np.ndarray, file: str = 'render.zip') -> None:
    with zipfile.ZipFile(file, 'w', compression=zipfile.ZIP_DEFLATED) as zp:
        zp.write('config.toml')
        zp.writestr('render.dat', data.tobytes())
