# %% [markdown]
# # Total variation exercise

# %%
import numpy as np
import skimage.io
from scipy.ndimage import gaussian_filter

# %% [markdown]
# Este ejercicio implementa el cálculo de la variación total de una imagen y compara este valor antes y después del suavizado.

# %%
# Function to compute total variation
def total_variation(im):
    '''
    Computes the total variation of an image.
    
    Parameters
    ----------
    im : ndarray
        Image.
    
    Returns
    -------
    float
        Total variation of the image.
    '''
    # calcula la suma de diferencia entre pixeles vecinos de las verticlaees + horizontales
    return np.abs(im[:-1] - im[1:]).sum() + np.abs(im[:,:-1] - im[:,1:]).sum()

# %%
# Read data
in_dir = './week1/'
im = skimage.io.imread(in_dir + 'fibres_xcth.png').astype(float)

# Compute total variation
tv_im = total_variation(im)

# Smooth the image
sigma = 2
im_g = gaussian_filter(im, sigma)
tv_im_g = total_variation(im_g)

# Print the results
print(f'Total variation of original image: {tv_im:0.4g}')
print(f'Total variation of smoothed image: {tv_im_g:0.4g}')
print(f'Total variation is reduced by: {tv_im-tv_im_g:0.4g}')    


# %% [markdown]
# La variación total de la imagen suavizada es menor que la de la imagen original
# La diferencia representa cuánto "ruido" o cambios bruscos se han eliminado
# Esto confirma que el suavizado gaussiano reduce efectivamente la variación total al suavizar transiciones bruscas


