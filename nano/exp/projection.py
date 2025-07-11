

#%%
%matplotlib tk

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
from functools import partial

np.random.seed(52)

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
from matplotlib.patches import ConnectionPatch
colors = np.random.rand(rand_3d_matrix.shape[0])
cmap = plt.get_cmap("viridis")
colors = cmap(np.linspace(0, 1, rand_3d_matrix.shape[0]))

fig, (axl, axr) = plt.subplots(1, 2, subplot_kw={'projection': '3d'}, figsize=(12,6)) 

colors_vec = ["blue", "red"]
def init(basis_vectors: np.ndarray, X: np.ndarray, proj_data: np.ndarray):
    axl.cla()
    axr.cla()
    axl.set_xlabel("x-axis")
    axl.set_ylabel("y-axis")
    axl.set_zlabel("z-axis")

    axl.set_xlim([0, np.max(X[:, 0])])
    axl.set_ylim([0, np.max(X[:, 1])])
    axl.set_zlim([0, np.max(X[:, 2])])
    axl.set_title('3D Vector Animation')

    for i in range(basis_vectors.shape[1]):
        axr.quiver(0,
                0,
                0,
                basis_vectors[:, i][0],
                basis_vectors[:, i][1],
                basis_vectors[:, i][2],
                    color=colors_vec[i],
                    arrow_length_ratio=0.2,
                    linestyles="solid")
        
    axr.set_xlabel("X Axis")
    axr.set_ylabel("Y Axis")
    axr.set_zlabel("Z Axis")
    axr.set_title('3D basis vectors')

    axr.set_xlim(0, np.max(projected_data[:,0]))
    axr.set_ylim(0,np.max(projected_data[:,1]))
    axr.set_zlim(0,1)
    return ()


def animate(frame, X: np.ndarray, proj_data: np.ndarray ):
    three_d_data = X[0:frame+1, :]
    two_d_proj_data = proj_data[0: frame+1, :]
    for j in range(frame):
        axl.quiver(0,0,0, three_d_data[j, 0], three_d_data[j, 1], three_d_data[j, 2],color=colors[j],
            arrow_length_ratio=0.1,
            linestyles="solid")
        
        axr.scatter(xs=two_d_proj_data[j, 0], ys = two_d_proj_data[j , 1], zs=0, c=colors[j], marker='x', label='2d projections')

basis_vectors = np.array([[1, 0], [0, 1], [0, 0]])
# basis_vectors = _orthonormal_basis(X=np.random.rand(3,2))
rand_3d_matrix = np.random.rand(100, 3)
projected_data = _low_dim_project(high_dim_data=rand_3d_matrix, basis_vectors=basis_vectors)

init_pars = partial(init, basis_vectors=basis_vectors, X=rand_3d_matrix, proj_data=projected_data)
animate_pars = partial(animate, X=rand_3d_matrix, proj_data=projected_data)

ani = FuncAnimation(fig=fig, func=animate_pars, frames=50, interval=300, init_func=init_pars)
plt.show()

# %%
