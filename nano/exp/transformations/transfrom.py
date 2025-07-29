from abc import ABC, abstractmethod
from typing import Any

from transformations.picture import Picture


class Transfrom(ABC):
    @abstractmethod
    def _build_transform_matrix(*args, **kwargs):
        raise NotImplementedError("to be implemented in the subclass")

    @abstractmethod
    def transform(obj: Picture, tranform_factor: Any):
        raise NotImplementedError("to be implemented in the subclass")
