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

    @property
    def original_dims(self):
        return self._original_img_dims

    @property
    def scaled_img_dims(self):
        return self._scaled_img_dim

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

    def _interploate(
        self, x_new: np.ndarray, y_new: np.ndarray, x_out: np.ndarray, y_out: np.ndarray
    ):
        return griddata(
            (x_new, y_new),
            self.img.flatten(),
            (x_out, y_out),
            method="linear",
        )

    def scale(self, x, y, x_out, y_out) -> Tuple[np.ndarray, np.ndarray]:
        flattend_grid = np.column_stack(
            (
                x.flatten(),
                y.flatten(),
                np.ones((x.shape[0] * x.shape[1],)),
            )
        )

        scaled_grids = np.dot(flattend_grid, self._scaling_matrix)
        x_new = scaled_grids[:, 0].astype(int)
        y_new = scaled_grids[:, 1].astype(int)

        return self._interploate(x_new, y_new, x_out, y_out)

    @staticmethod
    def plot(img: np.ndarray):
        plt.imshow(img, cmap="gray")


# %%

if __name__ == "__main__":
    transformer = ImageTransformer(img_path="./sample.jpg")
    ImageTransformer.plot(transformer.img)

    transformer.scaling_factor = (2, 2)

    transformer.create_scaling_matrix().define_dims()

    x, y = ImageTransformer.create_input_grids(transformer.original_dims)

    scaled_x, scaled_y = ImageTransformer.create_output_grids(
        transformer.scaled_img_dims
    )

    scaled_img = transformer.scale(x, y, scaled_x, scaled_y)
    ImageTransformer.plot(scaled_img)


# %%
