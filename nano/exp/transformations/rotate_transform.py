from typing import Tuple

import numpy as np
from scipy.interpolate import griddata
from transformations.picture import Dimensions, Picture
from transformations.transfrom import Transfrom

np.random.seed(52)


class RotateTransform(Transfrom):
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
            height=np.ceil(self.obj.dimensions.height * np.sqrt(2)).astype(int) + 1,
            width=np.ceil(self.obj.dimensions.width * np.sqrt(2)).astype(int) + 1,
        )

    @property
    def transform_factor(self):
        return self._transform_factor

    @transform_factor.setter
    def transform_factor(self, transform_factor: float):
        self._transform_factor = transform_factor

    @property
    def transform_matrix(self):
        return self._transform_matrix

    @staticmethod
    def create_grids(dims: Tuple) -> Tuple[np.ndarray, np.ndarray]:
        return np.meshgrid(np.arange(dims.width), np.arange(dims.height))

    def _build_transform_matrix(self):
        self._transform_matrix = np.array(
            [
                [
                    np.cos(self.transform_factor),
                    -np.sin(self.transform_factor),
                    0,
                ],
                [
                    np.sin(self.transform_factor),
                    np.cos(self.transform_factor),
                    0,
                ],
                [0, 0, 1],
            ]
        )

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
        x_new = scaled_grids[:, 0]
        y_new = scaled_grids[:, 1]

        x_min = np.min(x_new)
        y_min = np.min(y_new)

        x_new += -x_min
        y_new += -y_min

        return self._interploate(self.obj, x_new, y_new, x_out, y_out)

    def transform(self, tranform_factor):
        self.transform_factor = tranform_factor
        self.set_dimensions()
        self._build_transform_matrix()
        orig_x, orig_y = RotateTransform.create_grids(self.obj.dimensions)
        rot_x, rot_y = RotateTransform.create_grids(self.dimensions)
        return self._operate(orig_x, orig_y, rot_x, rot_y)
