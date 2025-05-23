# %% [markdown]
# # Boundary length exercise

# %%
# %% Boundary length exercise
import numpy as np
import skimage.io
import matplotlib.pyplot as plt

# Read data
in_dir = './week1/fuel_cells/'
imgs = [skimage.io.imread(in_dir + f'fuel_cell_{i}.tif') for i in range(1, 4)]
fig, ax = plt.subplots(1, 3, figsize=(15, 5))
for i in range(3):
    ax[i].imshow(imgs[i], cmap='gray')

# %% [markdown]
# Figure 1.7: Image of a segmented fuel
# cell with three phases. Black represents
# air, grey is the cathode, and white is the
# anode.

# %% [markdown]
# ### Task 1  
# La longitud del boundary es la medida de cuántos píxeles forman parte del contorno
# 
# Se calcula contando los pares de píxeles adyacentes que tienen valores diferentes
# 
# 
# 

# %%
# Just to show how it can be done, we solve the task by looping. It is better (more efficient, easier to read, more compact code) to use numpy vectorization as shown below.
for im in imgs:
    length = 0
    for i in range(im.shape[0]-1):
        for j in range(im.shape[1]):
            length += (im[i,j]!= im[i+1,j])
    for i in range(im.shape[0]):
        for j in range(im.shape[1]-1):
            length += (im[i,j]!= im[i,j+1]) 
    print(f'Boundary length: {length}')

# %% [markdown]
# ### Task 2 Vectorization
# 
# Método vectorizado (rápido):
# 
# En lugar de recorrer píxel por píxel, comparamos la imagen consigo misma, pero desplazada
# 
# Para bordes horizontales: Compara la imagen sin la última fila con la imagen sin la primera fila. Donde los valores difieren, hay un borde horizontal.
# 
# Para bordes verticales: Compara la imagen sin la última columna con la imagen sin la primera columna.
# 

# %%
#%% Section 1.1.2 - Task 2: Boundary length (vectorized)
for im in imgs:
    length = (im[:-1] != im[1:]).sum() + (im[:,:-1] != im[:,1:]).sum()
    print(f'Boundary length: {length}')

# %% [markdown]
# ### Task 3 Dedicated function

# %%
# %% Section 1.1.2 - Task 3: Boundary length function
def boundary_length(im):
    '''
    Computes the boundary length of an image.
    
    Parameters
    ----------
    im : ndarray
        Image.
        
    Returns
    -------
    bl : float
        Boundary length.
    '''
    return (im[:-1] != im[1:]).sum() + (im[:,:-1] != im[:,1:]).sum()


for im in imgs:
    print(f'Boundary length: {boundary_length(im)}')

for i in range(3):
    L = boundary_length(imgs[i])
    ax[i].imshow(imgs[i], cmap='gray')
    ax[i].set_title(f'L={L}')

# %%
# Esta técnica evita bucles lentos, aprovechando las operaciones optimizadas de NumPy para comparar arrays completos de una vez.


