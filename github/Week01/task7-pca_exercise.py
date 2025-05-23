# %% [markdown]
# # PCA exercise

# %%
from skimage.io import imread
import os
import matplotlib.pyplot as plt
import numpy as np

#%matplotlib tk

# %%
# Read data from spectral image
I = []
in_dir = './week1/mixed_green/'
im_names = sorted(os.listdir(in_dir))
images = [imread(in_dir + name) for name in im_names if name[-4:] == '.png']

# Show the images
fig, ax = plt.subplots(3, 6, figsize=(18, 6))
ax = ax.flatten()
for im, a in zip(images, ax):
    a.imshow(im, cmap='magma')

# %% [markdown]
# PCA CON NUMPY
# 

# %%
#%% PCA with numpy
I = np.stack(images, axis=-1)  # 960 x 1280 x 18
X = (I.reshape(-1, 18))  # Reshape to a matrix with 18 columns
mu = X.mean(axis=0)  # Compute the band-wise mean
X_mu = X - mu  # Subtract the mean
C = 1/(X.shape[0] - 1) * X_mu.T @ X_mu  # Compute the covariance matrix
l, V = np.linalg.eig(C)  # Compute the eigenvalues and eigenvectors
idx = l.argsort()[::-1] # Index of sorted eigenvalues - largest first
V = V[:, idx] # Sort the eigenvectors
Q = X_mu @ V # Compute the principal components
Q = Q.reshape(I.shape) # Reshape to image format

# Show the results
fig, ax = plt.subplots(3, 6, figsize=(18, 6))
ax = ax.flatten()
for i in range(18):
    ax[i].imshow(Q[:,:,i], cmap='magma')

# %% [markdown]
# PCA con python ya hecha
# 

# %%
import sklearn.decomposition

# %% Compare to PCA with sklearn
pca = sklearn.decomposition.PCA(n_components=18) # Create PCA object
pca.fit(X) # Fit the PCA object
Q = (pca.transform(X)).reshape(I.shape) # Compute the principal components

# Show the results (note that the sign of the eigenvectors can be flipped)
fig, ax = plt.subplots(3, 6, figsize=(18, 6))
ax = ax.flatten()
for i in range(18):
    ax[i].imshow(Q[:,:,i], cmap='magma')

# %% [markdown]
# El PCA (Análisis de Componentes Principales) para imágenes sirve principalmente para:
# 
# -   Reducción de dimensionalidad: Comprimir información de múltiples bandas espectrales en menos componentes.
# -   Extracción de características: Identificar los patrones más importantes de variación entre las imágenes.
# -   Eliminación de ruido: Separar la señal (componentes principales) del ruido (componentes menores).
# -   Visualización: Representar datos multiespectrales de forma comprensible.


