from collections import namedtuple
from typing import Tuple

import numpy as np
from scipy.interpolate import griddata
from transformations.picture import Dimensions, Picture
from transformations.transfrom import Transfrom

np.random.seed(52)


class ScalingTransform(Transfrom):
    TransFactor = namedtuple("TransFactor", ["x", "y"])

    def __init__(self, obj: Picture):
        self.obj = obj
        self._dimensions = Dimensions(height=0, width=0)
        self._transform_factor = None
        self._transform_matrix = None

    @property
    def dimensions(self) -> Tuple:
        return self._dimensions

    def set_dimensions(self):
        self._dimensions = Dimensions(
            height=(self.obj.dimensions.height * self.transform_factor.x),
            width=(self.obj.dimensions.width * self.transform_factor.y),
        )

    @property
    def transform_factor(self):
        return self._transform_factor

    @transform_factor.setter
    def transform_factor(self, transform_factor: Tuple[int, int]):
        self._transform_factor = ScalingTransform.TransFactor(
            x=transform_factor[0], y=transform_factor[1]
        )

    @property
    def transform_matrix(self):
        return self._transform_matrix

    def _build_transform_matrix(self):
        self._transform_matrix = np.array(
            [
                [self.transform_factor.x, 0, 0],
                [0, self.transform_factor.y, 0],
                [0, 0, 1],
            ]
        )

    @staticmethod
    def create_grids(dims: Tuple) -> Tuple[np.ndarray, np.ndarray]:
        return np.meshgrid(np.arange(dims.height), np.arange(dims.width))

    def _interploate(
        self,
        obj: Picture,
        x_new: np.ndarray,
        y_new: np.ndarray,
        x_out: np.ndarray,
        y_out: np.ndarray,
    ):
        return griddata(
            (x_new, y_new),
            obj.img.flatten(),
            (x_out, y_out),
            method="linear",
        )

    def _operate(
        self,
        x: np.ndarray,
        y: np.ndarray,
        x_out: np.ndarray,
        y_out: np.ndarray,
    ):
        flattend_grid = np.column_stack(
            (
                x.flatten(),
                y.flatten(),
                np.ones((x.shape[0] * x.shape[1],)),
            )
        )

        scaled_grids = np.dot(flattend_grid, self.transform_matrix)
        x_new = scaled_grids[:, 0].astype(int)
        y_new = scaled_grids[:, 1].astype(int)

        return self._interploate(self.obj, x_new, y_new, x_out, y_out)

    def transform(self, transform_factor: Tuple) -> np.ndarray:
        self.transform_factor = transform_factor
        self.set_dimensions()
        self._build_transform_matrix()
        orig_x, orig_y = ScalingTransform.create_grids(dims=self.obj.dimensions)
        scaled_x, scaled_y = ScalingTransform.create_grids(self.dimensions)
        return self._operate(orig_x, orig_y, scaled_x, scaled_y)
