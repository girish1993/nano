

#%%
%matplotlib tk

import os
import sys
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(dir_path)
parent_dir = os.path.dirname(dir_path)
sys.path.append(parent_dir)
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D

np.random.seed(52)

rand_3d_matrix = np.random.rand(100, 3)
basis_vectors = np.array([[1, 0], [0, 1], [0, 0]])
projection_matrix = np.dot(rand_3d_matrix, basis_vectors)




#%%
def _check_for_independence(X: np.ndarray) -> bool:
    return all(np.cross(X[:, 0], X[:, 1]) != np.zeros(3))


def _low_dim_project(high_dim_data: np.ndarray, basis_vectors: np.ndarray) -> np.ndarray:
    if high_dim_data.shape[1] != basis_vectors.shape[0]:
        raise ValueError("Cannot project with unequal dims")
    
    return np.dot(high_dim_data, basis_vectors)

def _orthonormal_basis(X: np.ndarray) -> np.ndarray:
    # this is just for (3,2)

    if _check_for_independence:
        u1 = X[:, 0] / np.linalg.norm(X[:, 0])
        u2 = X[:, 1] - ((np.dot(X[:, 1], u1)) * u1)

        u2 = u2 / np.linalg.norm(u2)
        return np.array([u1, u2]).T

    raise ValueError("Cannot orthonormalise dependent vectors")


#%%

# logic to animate the 3D plot

colors = np.random.rand(rand_3d_matrix.shape[0])
cmap = plt.get_cmap("viridis")
colors = cmap(np.linspace(0, 1, rand_3d_matrix.shape[0]))

fig, (axl, axr) = plt.subplots(1, 2, subplot_kw={'projection': '3d'}) 

# axl.set_aspect(1)
# axr.set_box_aspect(1 / 3)
# axr.yaxis.set_visible(False)

# fig = plt.figure()
# axl= fig.add_subplot(111, projection="3d")


def init():
    axl.cla()
    axl.set_xlabel("x-axis")
    axl.set_ylabel("y-axis")
    axl.set_zlabel("z-axis")

    axl.set_xlim([0, np.max(rand_3d_matrix[:, 0])])
    axl.set_ylim([0, np.max(rand_3d_matrix[:, 1])])
    axl.set_zlim([0, np.max(rand_3d_matrix[:, 2])])
    axl.set_title('3D Vector Animation')
    return ()


def animate(frame):

    data = rand_3d_matrix[0:frame+1, :]
    for j in range(frame):
        axl.quiver(0,0,0, data[j, 0], data[j, 1], data[j, 2],color=colors[j],
            arrow_length_ratio=0.1,
            linestyles="solid")
        

ani = FuncAnimation(fig=fig, func=animate, frames=50, interval=100, init_func=init)

plt.show()

# %%




fig =  plt.figure(figsize=(12, 6))

axl = fig.add_subplot(1, 2, 1 , projection='3d')



# basis_vectors = np.array([[1, 0], [0, 1], [0, 0]])
basis_vectors = _orthonormal_basis(np.random.rand(3, 2))

projected_data = _low_dim_project(high_dim_data=rand_3d_matrix, basis_vectors=basis_vectors)

colors = ["blue", "red"]
cmap = plt.get_cmap("viridis")
colors_scat = cmap(np.linspace(0, 1, rand_3d_matrix.shape[0]))

for i in range(basis_vectors.shape[1]):
    axl.quiver(0,
              0,
              0,
              basis_vectors[:, i][0],
            basis_vectors[:, i][1],
              basis_vectors[:, i][2],
                color=colors[i],
                  arrow_length_ratio=0.2,
                  linestyles="solid")

axl.scatter(xs=projected_data[:, 0], ys = projected_data[:, 1], zs=np.zeros(rand_3d_matrix.shape[0]), c=colors_scat, marker='x', label='2d projections')

axl.set_xlabel("X Axis")
axl.set_ylabel("Y Axis")
axl.set_zlabel("Z Axis")
axl.set_title('3D basis vectors')

axl.set_xlim(0, np.max(projected_data[:,0]))
axl.set_ylim(0,np.max(projected_data[:,1]))
axl.set_zlim(0,1)
plt.show()


# %%
