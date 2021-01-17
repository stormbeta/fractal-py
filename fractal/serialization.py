import pkg_resources
from datetime import datetime
from zipfile import ZipFile, ZIP_DEFLATED

from .colors import *
from .common import config, frame_params

fast_png: bool = 'lycon' in {pkg.key for pkg in pkg_resources.working_set}

if fast_png:
    import lycon
else:
    import png


def save_histogram_png(histdata):
    resolution = config.global_resolution
    output_filename = f"histogram/histogram{int(datetime.now().timestamp())}.png"
    histdata = (255/np.max(histdata) * histdata).astype('uint8')
    if fast_png:
        # Lycon is considerably faster than pypng, especially at larger resolutions where pypng is ridiculously slow
        # But it also kind of assumes GNU toolchain, and has dependencies on libpng-dev + libjpeg-dev
        lycon.save(output_filename, histdata.reshape(resolution, resolution))
    else:
        with open(output_filename, "wb") as fp:
            writer = png.Writer(resolution, resolution, greyscale=True)
            writer.write(fp, histdata.reshape(resolution, resolution))


def save_render_png(output, number: int = -1):
    resolution = config.global_resolution
    if number == -1:
        output_filename = f"{frame_params.folder}/nebula-{int(datetime.now().timestamp())}.png"
    else:
        output_filename = f"{frame_params.folder}/frame-{number:04d}.png"
    if fast_png:
        # Lycon is considerably faster than pypng, especially at larger resolutions where pypng is ridiculously slow
        # But it also kind of assumes GNU toolchain, and has dependencies on libpng-dev + libjpeg-dev
        lycon.save(output_filename, output.astype('uint8').reshape(resolution, resolution, 3))
    else:
        with open(output_filename, "wb") as fp:
            writer = png.Writer(resolution, resolution, greyscale=False)
            writer.write(fp, output.astype('uint8').reshape(resolution, resolution * 3))


def load_render_dat(file: str = 'render.zip') -> np.ndarray:
    with ZipFile(file, 'r') as zipFile:
        # Mutating global state as a side effect like this is kind of hacky, but config _is_ meant to be global
        config.reload(zipFile.open('config.toml').read())
        data = np.frombuffer(zipFile.open('render.dat').read(), dtype=np.float32)
        data.shape = config.rshape()
        return data.copy()


def save_render_dat(data: np.ndarray, file: str = 'render.zip') -> None:
    with ZipFile(file, 'w', compression=ZIP_DEFLATED) as zipFile:
        # NOTE: Minor bug - config.toml is saved at end of render, even though it might have been modified by user during render
        zipFile.write('config.toml')
        zipFile.writestr('render.dat', data.tobytes())
