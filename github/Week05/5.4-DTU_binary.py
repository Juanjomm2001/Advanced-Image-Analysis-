# %% [markdown]
# # Binary segmentation of DTU image

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
    centers = 0.5*(edges[:-1] + edges[1:]);
    for k in range(S.max()+1):
        ax.plot(centers, np.histogram(I[S==k].ravel(), edges)[0])

# %%
# noisy image
I = imread('./week5/DTU_noisy.png').astype(float)/255

# MRF parameters
beta  = 0.1
mu = [90/255, 170/255]

# mu=(μ₁,μ2)
# μ₁ = 90/255 ≈ 0.353 (clase oscura/fondo)
# μ₂ = 170/255 ≈ 0.667 (clase brillante/logo)





# Setting up graph with internal and external edges

# Interpretación: Píxel va a clase 0 si está más cerca de μ₀, a clase 1 si está más cerca de μ₁


g = maxflow.Graph[float]()
nodeids = g.add_grid_nodes(I.shape)
g.add_grid_edges(nodeids, beta)
g.add_grid_tedges(nodeids, (I - mu[1])**2, (I - mu[0])**2)

#  Graph cut
g.maxflow()
S = g.get_grid_segments(nodeids)

# Visualization
fig, ax = plt.subplots(1, 3)
ax[0].imshow(I, vmin=0, vmax=1, cmap=plt.cm.gray)
ax[0].set_title('Noisy image')
ax[1].imshow(S)
ax[1].set_title('Segmented')
segmentation_histogram(ax[2], I, S, edges=None)
ax[2].set_aspect(1./ax[2].get_data_ratio())
ax[2].set_title('Segmentation histogram')
plt.tight_layout()
plt.show()



# %%
import maxflow
print(maxflow.__file__)
print(dir(maxflow))

# %% [markdown]
# Efectos Esperados del Parámetro β
# β = 0.01 (muy bajo):
# 
# Segmentación ruidosa, similar a maximum likelihood
# Muchos píxeles aislados mal clasificados
# 
# β = 0.1 (bajo):
# 
# Equilibrio razonable
# Elimina algo de ruido manteniendo detalles
# 
# β = 1.0 (alto):
# 
# Segmentación muy suave
# Puede perder detalles finos del logo


