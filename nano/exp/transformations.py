# %%
from collections import namedtuple
from typing import Tuple

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import griddata

# %%

np.random.seed(52)

Scaling_Factor = namedtuple("Scaling_Factor", ["x", "y"])
Dimensions = namedtuple("Dimensions", ["height", "width"])


class ImageTransformer:
    def __init__(self, img_path: str) -> None:
        self.img = mpimg.imread(img_path)
        self._scaling_factor: Tuple = None
        self._scaling_matrix: np.ndarray = None
        self._original_img_dims: Tuple = None
        self._scaled_img_dim: Tuple = None

    @property
    def scaling_factor(self):
        return self._scaling_factor

    @scaling_factor.setter
    def scaling_factor(self, factor: Tuple):
        self._scaling_factor = Scaling_Factor(x=factor[0], y=factor[1])

    def create_scaling_matrix(self):
        self._scaling_matrix = np.array(
            [[self.scaling_factor.x, 0, 0], [0, self.scaling_factor.y, 0], [0, 0, 1]]
        )

        return self

    def define_dims(self):
        self._original_img_dims = Dimensions(
            height=self.img.shape[0], width=self.img.shape[1]
        )

        self._scaled_img_dim = Dimensions(
            height=self.img.shape[0] * self.scaling_factor.x,
            width=self.img.shape[1] * self.scaling_factor.y,
        )

        return self

    @staticmethod
    def create_input_grids(orignal_dims: Tuple) -> Tuple[np.ndarray, np.ndarray]:
        return np.meshgrid(
            np.arange(orignal_dims.height), np.arange(orignal_dims.width)
        )

    @staticmethod
    def create_output_grids(scaled_img_dims: Tuple) -> Tuple[np.ndarray, np.ndarray]:
        return np.meshgrid(
            np.arange(scaled_img_dims.height), np.arange(scaled_img_dims.width)
        )

    def scale(self, input_grids) -> Tuple[np.ndarray, np.ndarray]:
        flattend_grid = np.column_stack(
            input_grids[0].flatten(),
            input_grids[1].flatten(),
            np.ones(input_grids[0].shape[0] * input_grids[1].shape[1]),
        )

        scaled_grids = np.dot(flattend_grid, self._scaling_matrix)

        return scaled_grids[:, 0].astype(int), scaled_grids[:, 1].astype(int)

    def interploate(self, x: np.ndarray, y: np.ndarray, output_grid: Tuple):
        return griddata(
            (x, y),
            self.img.flatten(),
            (output_grid[0], output_grid[1]),
            method="linear",
        )

    @staticmethod
    def plot(img: np.ndarray):
        plt.imshow(img, cmap="gray")
