# %% [markdown]
# # Image unwrapping exercise

# %%
import numpy as np
import skimage.io
import matplotlib.pyplot as plt
import scipy.interpolate

# %% [markdown]
# Este ejercicio implementa la transformación de "unwrapping" (desenvolvimiento) de una imagen, convirtiendo una imagen en coordenadas cartesianas a coordenadas polares.

# %%
# Function to unwrap an image
def unwrap_im(im, n_angles, n_rad, center=None):
    '''
    Unwraps an image.

    Parameters
    ----------
    im : ndarray
        Image to be unwrapped.
    n_angles : int
        Number of angles.
    n_rad : int
        Number of radii.
    center : ndarray, optional
        Center of the image. The default is None.

    Returns
    -------
    im_rad : ndarray
        Unwrapped image.

    '''
    if center is None: # If no center is given, use the center of the image
        center = np.array(im.shape)/2
    angles = np.linspace(0, 2 * np.pi, n_angles)
    n_rad = int(min(min(im.shape)/2, n_rad))
    rad = np.linspace(0, n_rad, n_rad)
    # Create a grid of sampling points for X and Y
    X = np.outer(rad, np.cos(angles)) + center[0]
    Y = np.outer(rad, np.sin(angles)) + center[1]
    # Image grid for the interpolation object
    x = np.arange(0, im.shape[0])
    y = np.arange(0, im.shape[1])
    # Interpolation object
    f = scipy.interpolate.RectBivariateSpline(x, y, im)
    return f(X,Y,grid=False)

# %%
# Read data
in_dir = './week1/dental/'
im = skimage.io.imread(in_dir + 'slice105.png').astype(float)

# Show the original image
fig, ax = plt.subplots()
ax.imshow(im, cmap='gray')

# Unwrap the image
im_rad = unwrap_im(im, 200, 100)

# Show the results
fig, ax = plt.subplots()
ax.imshow(im_rad, cmap='gray')
plt.show()


# %% [markdown]
# Implementación específica:
# 
# La matriz resultante tiene dimensiones [n_rad, n_angles]
# Cada fila representa un radio específico
# Cada columna representa un ángulo específico
# El resultado es una representación "desarrollada" del objeto circular, donde:
# 
# El eje horizontal representa la dirección angular (0 a 2π)
# El eje vertical representa la distancia desde el centro
# 
# 
# 
# Esta transformación es particularmente útil para analizar estructuras con simetría circular o radial, como los implantes dentales mostrados en la Figura 1.11.


