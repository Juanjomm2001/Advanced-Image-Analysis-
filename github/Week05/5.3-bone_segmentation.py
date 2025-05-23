# %% [markdown]
# # Bone segmentation 

# %% [markdown]
# ¿Qué son los Graph Cuts?
# Los Graph Cuts son un algoritmo eficiente para encontrar la segmentación óptima (que minimiza la energía) en problemas MRF binarios.

# %% [markdown]
# Imports and helping functions

# %%
from skimage.io import imread
import matplotlib.pyplot as plt
import maxflow
import numpy as np

def segmentation_histogram(ax, I, S, edges=None):
    '''
    Histogram for data and each segmentation label.
    '''
    if edges is None:
        edges = np.linspace(I.min(), I.max(), 100)
    ax.hist(I.ravel(), bins=edges, color = 'k')
    centers = 0.5 * (edges[:-1] + edges[1:])
    for k in range(S.max() + 1):
        ax.plot(centers, np.histogram(I[S==k].ravel(), edges)[0])


# %% [markdown]
# Inspect the data. 

# %%
I = imread('./week5/V12_10X_x502.png').astype(float)/(2**16-1)

fig, ax = plt.subplots(1, 2, figsize=(12, 6))
ax[0].imshow(I, cmap=plt.cm.gray)
ax[0].set_title('image')

edges = np.linspace(0, 1, 257)
ax[1].hist(I.ravel(), edges)
ax[1].set_xlabel('pixel values')
ax[1].set_ylabel('count')
ax[1].set_title('intensity histogram')
ax[1].set_aspect(0.5/ax[1].get_data_ratio())
plt.tight_layout()
plt.show()

# %% [markdown]
# Define the likelihood and check the max-likelihood solution (thresholding).

# %%
# 2. Segmentación por Maximum Likelihood


#%% Define likelihood
mu = [0.40, 0.71]  #los dos puntos mas altos del histograma que sera ka media de las clases 
U = np.stack([(I-mu[i])**2 for i in range(len(mu))], axis=2) #asigna cada pixel a la media mas cercana 
S0 = np.argmin(U, axis=2)  #Resultado S0: Segmentación basada solo en intensidades (sin considerar vecinos)


fig, ax = plt.subplots(1, 2, figsize=(12, 6))
ax[0].imshow(S0)
ax[0].set_title('max likelihood')

segmentation_histogram(ax[1], I, S0, edges=edges)
ax[1].set_aspect(0.5/ax[1].get_data_ratio())
ax[1].set_xlabel('pixel values')
ax[1].set_ylabel('count')
ax[1].set_title('segmentation histogram')
plt.tight_layout()
plt.show()

# %% [markdown]
# 3. Segmentación con Graph Cuts (MRF)
# 
# Define MRF and use graph cuts to get the solution with the smallest posterior energy.

# %%
#%% Define prior, construct graph, solve
beta  = 0.1
g = maxflow.Graph[float]()
nodeids = g.add_grid_nodes(I.shape)
g.add_grid_edges(nodeids, beta)
g.add_grid_tedges(nodeids, U[...,1], U[...,0])

#  solving
g.maxflow()
S = g.get_grid_segments(nodeids)

fig, ax = plt.subplots(1, 2, figsize=(12, 6))
ax[0].imshow(S)
ax[0].set_title('max posterior')

segmentation_histogram(ax[1], I, S, edges=edges)
ax[1].set_aspect(0.5/ax[1].get_data_ratio())
ax[1].set_xlabel('pixel values')
ax[1].set_ylabel('count')
ax[1].set_title('segmentation histogram')
plt.tight_layout()
plt.show()

# %% [markdown]
# ## Optional: three classes
# 
# Use alpha expansion from maxflow to obtain three-class segmentation.

# %%
#%% Define likelihood
mu = [0.40, 0.46, 0.71]
U = np.stack([(I-mu[i])**2 for i in range(len(mu))], axis=2)
S0 = np.argmin(U, axis=2)

fig, ax = plt.subplots(1, 2, figsize=(12, 6))
ax[0].imshow(S0)
ax[0].set_title('max likelihood')

segmentation_histogram(ax[1], I, S0, edges=edges)
ax[1].set_aspect(0.5/ax[1].get_data_ratio())
ax[1].set_xlabel('pixel values')
ax[1].set_ylabel('count')
ax[1].set_title('segmentation histogram')
plt.tight_layout()
plt.show()

# %%
#%% Define prior and solve
beta  = 0.02
B = beta - beta * np.eye(len(mu))
#  solving
S = S0.copy()
maxflow.fastmin.aexpansion_grid(U, B, labels = S)


fig, ax = plt.subplots(1, 2, figsize=(12, 6))
ax[0].imshow(S)
ax[0].set_title('max posterior')

segmentation_histogram(ax[1], I, S, edges=edges)
ax[1].set_aspect(0.5/ax[1].get_data_ratio())
ax[1].set_xlabel('pixel values')
ax[1].set_ylabel('count')
ax[1].set_title('segmentation histogram')
plt.tight_layout()
plt.show()

# %%



