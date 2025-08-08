from collections import namedtuple
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

Dimensions = namedtuple("Dimensions", ["height", "width"])


class Picture:
    def __init__(self, img: Optional[np.ndarray] = None):
        self._img = img
        self._dimensions = (
            Dimensions(height=img.shape[0], width=img.shape[1])
            if img is not None
            else Dimensions(height=0, width=0)
        )

    @property
    def dimensions(self) -> Tuple:
        return self._dimensions

    @dimensions.setter
    def dimensions(self, dims: Tuple):
        self._dimensions = Dimensions(height=dims[0], width=dims[1])

    @property
    def img(self):
        return self._img

    def read_image(self, img_path: str):
        self._img = np.array(Image.open(img_path))
        self.dimensions = self._img.shape

    @staticmethod
    def plot(img: np.ndarray, title: Optional[str] = "image plt"):
        plt.figure()
        plt.imshow(img, cmap="gray")
        plt.title(title)
        plt.show()
