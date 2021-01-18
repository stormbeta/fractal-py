import logging
import sys
import time
import math

from PyQt5 import QtGui, QtCore, QtWidgets
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QFormLayout, QGridLayout, QComboBox

from typing import List, Callable
from fractal.common import config
import numpy as np
from . import serialization
from .colors import colorize_simple2, scalers


logging.basicConfig(level=config.log_level, format="%(message)s")
log = logging.getLogger('gui')


# TODO: Support rendering from GUI, display progress in window somehow
# TODO: Support re-rendering with different parameters (only works for fixed values, maybe limited template)


"""
Poor attempt at creating a basic UI frontend, which I'm not very good at
For now, only handles colorization since that's fast enough to do in real-time (ish, the code isn't terribly optimized, and it's pretty slow past 2048x2048)
"""

class Slider(QtWidgets.QSlider):
    fractal: QWidget
    # TODO: Should probably be an enum
    data_channel: int
    color_channel: int

    def __init__(self, fractal, data_channel: int, color_channel: int):
        super().__init__()
        self.setTracking(True)
        self.data_channel = data_channel
        self.color_channel = color_channel
        self.fractal = fractal
        self.setValue(fractal.colorScale[self.color_channel][self.data_channel])
        self.setMinimum(0)
        self.setMaximum(config.color_scale_max)
        self.setGeometry(0, 0, 256, 32)
        self.setOrientation(Qt.Horizontal)
        # NOTE: I'm not sure why I can't just override this method via inheritance?
        self.valueChanged.connect(self.recolor)

    def recolor(self, value: int) -> None:
        startTime = time.time_ns()
        self.fractal.colorScale[self.color_channel][self.data_channel] = value
        self.fractal.colorize()
        log.debug(f"Recolor: {(time.time_ns() - startTime)/1000000:0.0f}ms")


class FractalApp(QWidget):
    buffer: np.ndarray
    render_view: QLabel
    resolution: int
    colorScale: List[List[int]]
    colorFunctions = List[QComboBox]

    def __init__(self):
        super().__init__()
        self.grid = QGridLayout(self)

        self.colorScale = config.color_scale
        self.data = serialization.load_render_dat()
        self.resolution = config.global_resolution
        self.buffer = np.full(dtype=np.uint8, fill_value=128, shape=(self.resolution, self.resolution, 3))

        self.render_view = QLabel(self)
        self.render_view.setGeometry(QtCore.QRect(0, 0, 1024, 1024))
        self.grid.addWidget(self.render_view, 0, 0)
        self.layout = QFormLayout(self)
        self.grid.addLayout(self.layout, 0, 1)
        self.layout.setGeometry(QtCore.QRect(1024, 0, 256, 1024))

        self.initSliders()
        self.colorize()

        self.setGeometry(256, 0, 1024 + 256, 1024)
        self.render_view.show()
        self.show()

    def initSliders(self):
        i: int = 0
        self.colorFunctions = []
        for c in ['red', 'green', 'blue']:
            scaler = QComboBox()
            self.colorFunctions.append(scaler)
            for name, func in scalers.items():
                scaler.addItem(name, name)
            scaler.setCurrentText(config.color_algo[i])
            self.layout.addRow(QLabel(c), scaler)
            for s in [(0, "n"), (1, "i"), (2, "θ")]:
                slider = Slider(self, s[0], i)
                label = QLabel(f"{s[1]}")
                self.layout.addRow(label, slider)
                # slider.setGeometry(0, save_button.height() + slider.height()*s, slider.width(), slider.height())
                slider.show()
            i += 1
            scaler.currentIndexChanged.connect(self.colorize)

    def addWidget(self, widget: QWidget) -> None:
        self.layout.addWidget(widget)

    def logColorScale(self):
        print(self.colorScale)

    def save(self):
        log.info(self.colorScale)
        serialization.save_render_png(self.buffer)

    def colorize(self):
        data = self.data.copy()
        sf = [scalers[sel.currentData()] for sel in self.colorFunctions]
        data[...] = colorize_simple2(data, self.colorScale, sf)
        self.buffer = np.minimum(255, data).astype(np.uint8)
        img_buffer = QImage(self.buffer, self.resolution, self.resolution, QImage.Format_RGB888)
        pix = QPixmap.fromImage(img_buffer)
        self.render_view.setPixmap(pix.scaled(1024, 1024))


def run_app():
    app = QApplication(sys.argv)
    main = FractalApp()

    save_button = QtWidgets.QPushButton()
    save_button.setText("SAVE")
    save_button.show()
    save_button.pressed.connect(main.save)

    color_button = QtWidgets.QPushButton()
    color_button.setText("colorLog")
    color_button.show()
    color_button.pressed.connect(main.logColorScale)
    main.layout.addWidget(save_button)
    main.layout.addWidget(color_button)
    sys.exit(app.exec_())

if __name__ == '__main__':
    run_app()
