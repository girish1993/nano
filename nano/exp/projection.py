# %%

import os
import sys

dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(dir_path)

parent_dir = os.path.dirname(dir_path)
sys.path.append(parent_dir)

import matplotlib.cm as cm  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from base.matrix import Matrix  # noqa: E402
from base.vector import Vector  # noqa: E402

rand_3d_matrix = np.random.rand(100, 3)
basis_vectors = np.array([[1, 0], [0, 1], [0, 0]])

projection_matrix = np.dot(rand_3d_matrix, basis_vectors)
print(projection_matrix)


# %%
# matrix and vector implementation

M = Matrix(list([Vector(row) for row in rand_3d_matrix]))
b = Matrix(list([Vector(row) for row in basis_vectors]))

projection_matrix = M * b

print(projection_matrix)

# %%
# Starting point (origin)
x0, y0, z0 = 0, 0, 0

fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")


# %%

fig = plt.figure()


def _plot_vectors(mat: np.ndarray, basis: np.ndarray):
    ax = fig.add_subplot(111, projection="3d")
    colors = np.random.rand(rand_3d_matrix.shape[0])
    cmap = cm.get_cmap("Paired")
    colors = cmap(np.linspace(0, 1, rand_3d_matrix.shape[0]))
    for i in range(len(rand_3d_matrix)):
        ax.quiver(
            0,
            0,
            0,
            rand_3d_matrix[i, 0],
            rand_3d_matrix[i, 1],
            rand_3d_matrix[i, 2],
            color=colors[i],
            linestyle="-",
            linewidth=1.5,
        )

    ax.set_xlabel("X Axis")
    ax.set_ylabel("Y Axis")
    ax.set_zlabel("Z Axis")

    ax.set_xlim([0, np.max(rand_3d_matrix[:, 0])])
    ax.set_ylim([0, np.max(rand_3d_matrix[:, 1])])
    ax.set_zlim([0, np.max(rand_3d_matrix[:, 2])])


def _check_for_independence(X: np.ndarray) -> bool:
    return all(np.cross(X[:, 0], X[:, 1]) != np.zeros(3))


def _orthonormal_basis(X: np.ndarray) -> np.ndarray:
    # this is just for (3,2)

    if _check_for_independence:
        u1 = r_basis[:, 0] / np.linalg.norm(r_basis[:, 0])
        u2 = r_basis[:, 1] - ((np.dot(r_basis[:, 1], u1)) * u1)

        u2 = u2 / np.linalg.norm(u2)
        return np.array([u1, u2]).T

    raise ValueError("Cannot orthonormalise dependent vectors")


r_basis = np.random.rand(3, 2)
basis = _orthonormal_basis(X=r_basis)
_plot_vectors(mat=rand_3d_matrix, basis=basis)

# %%
