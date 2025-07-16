# %%
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np

np.random.seed(52)

# %%

img = mpimg.imread("./sample.jpg")

plt.imshow(img, cmap="gray")

# %%

# scaling
scaling_matrix = np.zeros((3,3))
scaling_matrix[0,0] = 2
scaling_matrix[1,1] = 2
scaling_matrix[2,2] = 1

scaled_img_dims = (int(img.shape[0] * scaling_matrix[0,0]), int(img.shape[1] *scaling_matrix[1,1]))


scaled_img = np.zeros(scaled_img_dims)



X, Y = np.meshgrid(np.arange(img.shape[0]), np.arange(img.shape[1]))

flattend_grid = np.column_stack((X.flatten(), Y.flatten(), np.ones(X.shape[0] * X.shape[1])))

scaled_grids = np.dot(flattend_grid, scaling_matrix)



# %%
