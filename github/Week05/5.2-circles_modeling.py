# %% [markdown]
# Los priors geométricos son información adicional sobre la geometría esperada de los objetos que queremos segmentar. En lugar de analizar solo la intensidad de los píxeles, incorporamos conocimiento sobre:
# 
# Forma:
# Posición: 
# Tamaño: 
# Orientación:
# 
# - Métodos previos (como segmentación basada en características):
# 
#     Solo usan apariencia local (textura, intensidad)
#     No consideran forma o geometría global

# %% [markdown]
# # Segmentation modeling
# 
# ¿Qué es un Markov Random Field (MRF)?
# Un MRF es un modelo probabilístico que considera las dependencias espaciales entre píxeles vecinos. La idea clave es que la etiqueta de un píxel no depende solo de su intensidad, sino también de las etiquetas de sus vecinos.
# 
# 
# Problema clave: Como muestra el histograma (Figura 5.2), las distribuciones de intensidad se superponen. Un simple umbralizado fallaría porque píxeles de diferentes clases pueden tener intensidades similares.
# 
# - Sin MRF: Un píxel ruidoso podría clasificarse mal basándose solo en su intensidad.
# - Con MRF: Aunque un píxel tenga intensidad ambigua, si todos sus vecinos pertenecen a una clase específica, es probable que también pertenezca a esa clase.
# 
# 
# 

# %% [markdown]
# # Objetivo del Ejercicio
# Verificar que esta función de energía realmente funciona:
# 
# - Crear diferentes segmentaciones candidatas
# - Calcular la energía de cada una
# - Confirmar que segmentaciones mejores tienen menor energía
# - Demostrar que la segmentación ground truth tiene la energía más baja
# 
# Esto valida que minimizar la energía MRF conduce hacia la solución deseada.

# %%
import skimage.io
import numpy as np
import scipy.ndimage
import matplotlib.pyplot as plt


# %%
path = './week5/'
D = skimage.io.imread(path + 'noisy_circles.png').astype(float)

# Ground-truth segmentation.
GT = skimage.io.imread(path + 'noise_free_circles.png')
(mu, S_gt) = np.unique(GT, return_inverse=True) # # Extrae medias μ y segmentación GT

S_gt = S_gt.reshape(D.shape)

segmentations = [S_gt]  # list where I'll place different segmentations

# %% [markdown]
# You should write the code which computes segmentation energies (U1 and U2) as described in the lecture notes. The skeleton of the function is provided below. 

# %% [markdown]
# La energía es una puntuación que mide "qué tan buena" es una segmentación. Energía baja = buena segmentación. Es como un "coste" que queremos minimizar.

# %%
def segmentation_energy(S, I, mu, beta):
    #S imagen con etiquetas es decir en cada pixel la etiqueta de la región a la que pertenece
    #I imagen de intensidades
    #mu vector de medias de cada región es decir las intensidades qeu se esperan para la clase 1 ,2,3...





    # likelihood energy 
    # Likelihood: Σ(μ(fi) - di)²

    # U₁: ¿Qué tan bien clasifica según intensidades?

    U1 = ((mu[S] - I)**2).sum()   #Calcula V₁ (likelihood) como suma de diferencias cuadradas


    # U₂: ¿Qué tan suave es espacialmente?

    
    # prior energy
    U2 = beta * ((S[1:,:] != S[:-1,:]).sum() + (S[:,1:] != S[:,:-1]).sum())  #Calcula V₂ (prior) contando píxeles vecinos con etiquetas diferentes

    return U1, U2 






"""
U₁ bajo = píxeles bien clasificados según su intensidad
U₁ alto = muchos píxeles mal clasificados

U₂ bajo = regiones suaves, pocas discontinuidades
U₂ alto = muchos cambios entre píxeles vecinos (ruidoso)

beta es cuando se penaliza los bordes entre regiones, cuanto mas alto mas se penaliza 
β Pequeño (ej. β = 10): Resultado: Segmentación ruidosa pero fiel a los datos

β Balanceado (ej. β = 1000):

Equilibrio entre fidelidad a datos y suavidad
Resultado: Segmentación limpia y coherente


"""


# %% [markdown]
# A helping function for plotting the histograms of the data and the segmentation.

# %%
def segmentation_histogram(ax, D, S, edges=None):
    '''
    Plot histogram for grayscale data and each segmentation label.
    '''
    if edges is None:
        edges = np.linspace(D.min(), D.max(), 100)
    ax.hist(D.ravel(), bins=edges, color = 'k')
    centers = 0.5 * (edges[:-1] + edges[1:])
    for k in range(S.max() + 1):
        ax.plot(centers, np.histogram(D[S==k].ravel(), edges)[0])

# %% [markdown]
# Get hold of the data (noisy image) and ground truth segmentation.

# %% [markdown]
# Find some configurations (segmentations) using conventional segmentation methods.
# 

# %%
# PARA HACER VARIASSEGMENTACIONES DEL GROUNTHTRUTH Y VER y los mete a la lista de segmentaciones 


# Simple thresholding
S_t = np.zeros(D.shape, dtype=int) + (D > 100) + (D > 160) # thresholded
segmentations += [S_t]

# Gaussian filtering followed by thresholding
D_s = scipy.ndimage.gaussian_filter(D, sigma=1, truncate=3, mode='nearest')
S_g = np.zeros(D.shape, dtype=int) + (D_s > 100) + (D_s > 160) 
segmentations += [S_g]

# Median filtering followed by thresholding
D_m = scipy.ndimage.median_filter(D, size=(5, 5), mode='reflect')
S_t = np.zeros(D.shape, dtype=int) + (D_m > 100) + (D_m > 160) # thresholded
segmentations += [S_t]

# %% [markdown]
# Visualize the segmentations, associated histograms and error images. Once you implement the `segmentation_energy` you can look at whether there is a link between segmentation energies and the quality of the segmentation.
# 
# 

# %%
#%% visualization
fig, ax = plt.subplots()
ax.imshow(D, vmin=0, vmax=255, cmap=plt.cm.gray)  #muestra D normal 
plt.show()


fig, ax = plt.subplots(3, len(segmentations), figsize=(10, 10))
beta = 1000

for i, s in enumerate(segmentations):
    ax[0][i].imshow(s) 
    U1, U2 = segmentation_energy(s, D, mu, beta)
    ax[0][i].set_title(f'likelihood: {int(U1)}\nprior: {U2}\nposterior: {int(U1)+U2}')
    
    segmentation_histogram(ax[1][i], D, s)
    ax[1][i].set_xlabel('Intensity')
    ax[1][i].set_ylabel('Count')
    
    err = S_gt - s
    ax[2][i].imshow(err, vmin=-2, vmax=2, cmap=plt.cm.bwr)
    ax[2][i].set_title(f'Pixel error: {(err != 0).sum()}')

fig.tight_layout()
plt.show()   

# %%



