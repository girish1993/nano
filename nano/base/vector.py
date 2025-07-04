import os
import sys
from math import pow, sqrt
from typing import List, Union

dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(dir_path)

parent_dir = os.path.dirname(dir_path)
sys.path.append(parent_dir)


class Vector:
    def __init__(self, inp: Union[List[int], List[float]]) -> None:
        self._inp = inp
        self._size = len(inp)

    @property
    def size(self):
        return self._size

    @property
    def inp(self):
        return self._inp

    def __getitem__(self, index: int):
        return self.inp[index]

    def __setitem__(self, index: int, value: int | float):
        self.inp[index] = value

    def __add__(self, other_vector: "Vector") -> "Vector":
        if self.size != other_vector.size:
            raise ValueError("Vectors must of the same size")

        return Vector([x + y for x, y in zip(self.inp, other_vector.inp)])

    def __sub__(self, other_vector: "Vector") -> "Vector":
        if self.size != other_vector.size:
            raise ValueError("Vectors must of the same size")

        return Vector([x - y for x, y in zip(self.inp, other_vector.inp)])

    def __mul__(
        self, other: Union["Vector", (int | float)]
    ) -> Union["Vector", int, float]:
        if isinstance(other, (int, float)):
            return Vector([other * x for x in self._inp])

        if self.size != other.size:
            raise ValueError("vectors must be of the same size")

        return sum([x * y for x, y in zip(self.inp, other.inp)])

    def __rmul__(self, other: (int | float)) -> "Vector":
        return Vector([other * x for x in self._inp])

    def __str__(self) -> str:
        return f"{self.inp}"

    def __len__(self) -> int:
        return self._size

    def __round__(self, precision: int) -> "Vector":
        return Vector([round(x, precision) for x in self.inp])

    def magnitude(self) -> float:
        return sqrt(sum([pow(elem, 2) for elem in self.inp]))


if __name__ == "__main__":
    try:
        v1 = Vector([1, 2, 3])
        v2 = Vector([2.5566, 3.666, 4.5555])
        x = -10

        print(round(v2, 2))
        # print(((v1 * x) - v2))
    except:
        raise
