from abc import ABC, abstractmethod
from collections import namedtuple
from typing import Any, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy.interpolate import griddata

np.random.seed(52)

Dimensions = namedtuple("Dimensions", ["height", "width"])


class Picture:
    def __init__(self):
        self._dimensions = Dimensions(height=0, width=0)
        self._img = None

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
        plt.imshow(img, cmap="gray")
        plt.title(title)
        plt.show()


class Transfrom(ABC):
    @abstractmethod
    def _build_transform_matrix(*args, **kwargs):
        raise NotImplementedError("to be implemented in the subclass")

    @abstractmethod
    def transform(obj: Picture, tranform_factor: Any):
        raise NotImplementedError("to be implemented in the subclass")


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


class ImageTransformer:
    _TRANSFROM_MAPPING = {
        "scale": ScalingTransform,
    }

    def __init__(
        self, obj: Picture, transform_type: str, tranform_factor: Tuple[int, int]
    ) -> None:
        self.transform_instance = ImageTransformer._TRANSFROM_MAPPING.get(
            transform_type
        )(obj)
        self.transform_factor = tranform_factor

    def transform(self):
        return self.transform_instance.transform(self.transform_factor)

    def create_tranform_matrix(self, type: str):
        if type == "scale":
            self._transform_matrix = np.array(
                [
                    [self.transform_factor.x, 0, 0],
                    [0, self.transform_factor.y, 0],
                    [0, 0, 1],
                ]
            )
        elif type == "rotate":
            self._transform_matrix = np.array(
                [
                    [
                        np.cos(self.transform_factor.theta),
                        -np.sin(self.transform_factor.theta),
                        0,
                    ],
                    [
                        np.sin(self.transform_factor.theta),
                        np.cos(self.transform_factor.theta),
                        0,
                    ],
                    [0, 0, 1],
                ]
            )

        return self


if __name__ == "__main__":
    pic = Picture()
    pic.read_image(img_path="nano/assets/sample.jpg")

    # Picture.plot(img=pic.img)

    img_trnsfrmr = ImageTransformer(
        obj=pic, transform_type="scale", tranform_factor=(4, 1.5)
    )

    transfromed_img = img_trnsfrmr.transform()

    Picture.plot(img=transfromed_img)
