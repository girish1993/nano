import copy
import itertools
from collections import namedtuple
from functools import reduce
from typing import List, Tuple, Union

from vector import Vector


class Matrix:
    lu_decomp_parts = namedtuple("LUDecompParts", ["L", "U"])

    def __init__(self, rows: List[List[int | float] | float | Vector]):
        self._rows = Matrix._set_rows(rows=rows)
        self._row_count = len(rows) if rows else 0
        self._col_count = Matrix._get_col_count(rows=rows) if rows else 0

    @property
    def rows(self):
        return self._rows

    @property
    def row_count(self):
        return self._row_count

    @property
    def col_count(self):
        return self._col_count

    @property
    def T(self) -> "Matrix":
        if self.row_count == self.col_count:
            mat_t: Matrix = Matrix(copy.deepcopy(self.rows))
            for i in range(mat_t.row_count):
                for j in range(i + 1, mat_t.col_count):
                    mat_t.rows[i][j], mat_t.rows[j][i] = (
                        mat_t.rows[j][i],
                        mat_t.rows[i][j],
                    )
            return mat_t

        non_sq_mat_t: Matrix = Matrix([Vector([]) for _ in range(self.col_count)])
        for i, j in list(
            itertools.product(range(self.col_count), range(self.row_count))
        ):
            non_sq_mat_t.rows[i].inp.insert(j, self.rows[j][i])
        return non_sq_mat_t

    @property
    def shape(self) -> Tuple:
        return (self.row_count, self.col_count)

    @staticmethod
    def _set_rows(rows: List[List[int | float] | List[Vector]]):
        if not rows:
            return []

        if all(isinstance(row, list) for row in rows):
            return [Vector(list(map(float, row))) for row in rows]

        if all(isinstance(row, float) for row in rows):
            return rows

        if all(isinstance(row, Vector) for row in rows):
            return rows

    @staticmethod
    def _get_col_count(rows: List[Vector]) -> int:
        if Matrix._check_col_count(rows=rows):
            return len(rows[0])
        else:
            raise ValueError("Matrix should have same number of columns")

    @staticmethod
    def _check_col_count(rows: List[Vector]) -> bool:
        if len(set(list(map(len, rows)))) == 1:
            return True
        return False

    def __add__(self, other: "Matrix") -> "Matrix":
        if self.shape != other.shape:
            raise ValueError("Matrices should be of the same shape to be added")

        return Matrix([x + y for x, y in zip(self.rows, other.rows)])

    def __mul__(self, other: Union["Matrix", int, float]) -> "Matrix":
        if isinstance(other, Matrix):
            if self.col_count != other.row_count:
                raise ValueError(
                    "Cannot multiple matrices with unequal cols and row counts"
                )

            col_vecs = [
                Vector([row[i] for row in other.rows]) for i in range(other.col_count)
            ]

            product_rows = [
                Vector([row * col for col in col_vecs]) for row in self.rows
            ]

            return Matrix(product_rows)

        return Matrix([row * other for row in self.rows])

    def __rmul__(self, other: (int | float)):
        return Matrix([other * row for row in self.rows])

    def __sub__(self, other: "Matrix") -> "Matrix":
        if self.shape != other.shape:
            raise ValueError("Matrices should be of the same shape to be added")

        return Matrix([x - y for x, y in zip(self.rows, other.rows)])

    @classmethod
    def _create_identity_matrix(cls, num_rows: int) -> "Matrix":
        rows = []
        for i in range(num_rows):
            rows.append(Vector([1 if j == i else 0 for j in range(num_rows)]))
        return Matrix(rows)

    @classmethod
    def _gaus_elim(cls, mat: "Matrix") -> "Matrix":
        factors = {}
        id_mat = cls._create_identity_matrix(num_rows=mat.row_count)
        for i in range(mat.row_count - 1):
            factors = factors | {
                f"{i}_{j}": (mat.rows[j][i] / mat.rows[i][i]) if mat.rows[i][i] else 0
                for j in range(i + 1, mat.row_count)
            }

            for j in range(i + 1, mat.row_count):
                mat.rows[j] = round(
                    mat.rows[j] - (round((factors.get(f"{i}_{j}") * mat.rows[i]), 2)), 2
                )

        return mat

    @staticmethod
    def _get_det(mat: "Matrix") -> float:
        return round(
            reduce(
                lambda elem1, elem2: elem1 * elem2,
                [mat.rows[i][i] for i in range(mat.row_count)],
            ),
            2,
        )

    def det(self) -> float:
        if self.shape == (1, 1):
            raise ValueError("Not enough values to unpack for determinant calculation")

        if self.shape == (2, 2):
            return (self.rows[0][0] * self.rows[1][1]) - (
                self.rows[0][1] * self.rows[1][0]
            )

        if self.row_count != self.col_count:
            raise ValueError("Non-square matrix cannot have a determinant")

        ref_matrix: Matrix = Matrix._gaus_elim(mat=self)
        return Matrix._get_det(mat=ref_matrix)

    def __str__(self) -> str:
        return (
            (
                f"{self.__class__.__name__} : \n"
                + "["
                + "\n".join(str(row) for row in self.rows)
                + "]"
            )
            if self.rows
            else "[]"
        )


if __name__ == "__main__":
    try:
        m1 = Matrix([[3, 4, 4], [3, 6, 9], [3, 6, 8]])
        m2 = Matrix(
            [
                [7, 12, 3, 16, 9],
                [5, 18, 2, 14, 11],
                [13, 6, 19, 8, 4],
                [10, 15, 7, 17, 1],
                [4, 5, 6, 7, 7],
                [6, 7, 8, 9, 0],
            ]
        )
        print(m1)
        # np_arr = np.array(
        #     [
        #         [7, 12, 3, 16, 9],
        #         [5, 18, 2, 14, 11],
        #         [13, 6, 19, 8, 4],
        #         [10, 15, 7, 17, 1],
        #         [8, 2, 20, 5, 12],
        #     ]
        # )
        # determinant = np.linalg.det(np_arr)
        # print("Determinant:", determinant)
        # # print(m1 * m2)
    except:
        raise
