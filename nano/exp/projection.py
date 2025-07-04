# %%

import os
import sys

dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(dir_path)

parent_dir = os.path.dirname(dir_path)
sys.path.append(parent_dir)

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
import matplotlib.pyplot as plt

# Starting point (origin)
x0, y0, z0 = 0, 0, 0
# Vector components
u, v, w = 3, 4, 5

fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
colors = np.random.rand(rand_3d_matrix.shape[0])
ax.quiver(
    x0,
    y0,
    z0,
    rand_3d_matrix[:, 0],
    rand_3d_matrix[:, 1],
    rand_3d_matrix[:, 2],
    cmap="viridis",
    arrow_length_ratio=0.1,
)

ax.set_xlabel("X Axis")
ax.set_ylabel("Y Axis")
ax.set_zlabel("Z Axis")

ax.set_xlim([0, np.max(rand_3d_matrix[:, 0])])
ax.set_ylim([0, np.max(rand_3d_matrix[:, 1])])
ax.set_zlim([0, np.max(rand_3d_matrix[:, 2])])

plt.show()


# %%
