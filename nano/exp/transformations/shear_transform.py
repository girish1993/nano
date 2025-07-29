from collections import namedtuple
from typing import Tuple

import numpy as np
from scipy.interpolate import griddata
from transformations.picture import Dimensions, Picture
from transformations.transfrom import Transfrom


class ShearTransform(Transfrom):
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
            height=np.ceil(
                self.obj.dimensions.height
                + (np.abs(self.transform_factor.x) * self.obj.dimensions.height)
            ),
            width=np.ceil(
                self.obj.dimensions.width
                + (np.abs(self.transform_factor.y) * self.obj.dimensions.width)
            ),
        )

    @property
    def transform_factor(self):
        return self._transform_factor

    @transform_factor.setter
    def transform_factor(self, transform_factor: Tuple[float, float]):
        self._transform_factor = ShearTransform.TransFactor(
            x=transform_factor[0], y=transform_factor[1]
        )

    @property
    def transform_matrix(self):
        return self._transform_matrix

    def _build_transform_matrix(self):
        self._transform_matrix = np.array(
            [
                [1, self.transform_factor.x, 0],
                [self.transform_factor.y, 1, 0],
                [0, 0, 1],
            ]
        )

    @staticmethod
    def create_grids(dims: Tuple) -> Tuple[np.ndarray, np.ndarray]:
        return np.meshgrid(np.arange(dims.width), np.arange(dims.height))

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

    def transform(self, tranform_factor: Tuple[float, float]):
        self.transform_factor = tranform_factor
        self.set_dimensions()
        self._build_transform_matrix()
        orig_x, orig_y = ShearTransform.create_grids(dims=self.obj.dimensions)
        scaled_x, scaled_y = ShearTransform.create_grids(self.dimensions)
        return self._operate(orig_x, orig_y, scaled_x, scaled_y)
