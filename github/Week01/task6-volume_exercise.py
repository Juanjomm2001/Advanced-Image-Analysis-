# %% [markdown]
# # Volume exercise

# %%
import numpy as np
from skimage.io import imread
import matplotlib.pyplot as plt
import os
# %matplotlib tk

# %% [markdown]
# Esto esta guapisimo, junta todas las imagens que estan hechas desd earriba en un vector apilandolas en el eje z. luego puedes revisar los ejes y te dara la iamgen 3 d desde otdos los planos.

# %%
# Read data into volume
V = []
in_dir = './week1/dental/'
im_names = sorted(os.listdir(in_dir))
V = [imread(in_dir + n).astype(float) for n in im_names if n[-4:] == '.png'] #volumen 3D (un array 3D de NumPy) que se construye apilando todas las imágenes 2D (cortes o "slices") 

V = np.array(V) # Convert to numpy array
hist, bins = np.histogram(V, bins=np.arange(-0.5, 256.5, 1)) # muestra la distribución de intensidades de todos los píxeles del volumen

fig, ax = plt.subplots(2, 2)
ax[0, 0].bar(np.arange(0, 256), hist, width=1)
ax[1, 0].imshow(V[100], cmap='gray')  #Vista axial (plano XY): muestra el corte número 100 en Z. la misma al dataset
ax[0, 1].imshow(V[:, 125])  # Vista coronal DESDE ARRIBA (plano XZ): muestra una sección vertical en Y=125
ax[1, 1].imshow(V[:, :, 125], cmap='Pastel1') #Vista sagital (plano YZ): muestra una sección vertical en X=125

# Show thresholding
thresh = 175
fig, ax = plt.subplots(1, 3)
ax[0].imshow(V[100] > thresh, cmap='gray')
ax[1].imshow(V[:, 125] > thresh, cmap='gray')
ax[2].imshow(V[:, :, 125] > thresh, cmap='gray')

plt.show()




