# %% [markdown]
# # 1 Image convolution

# %%
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import convolve
from skimage.io import imread

# %% [markdown]
# ### Task 1 Create Gaussian kernel

# %% [markdown]
# ![image.png](attachment:image.png)

# %%
# ""
# 1. Create Gaussian kernel. Gaussian kernels are usually truncated at
# the value between 3 and 5 times σ. You can create a kernel as follows:


def get_gaussian_kernels(s):
    '''
    Returns a 1D Gaussian kernel and its derivative and the x values.
    
    Parameters
    ----------
    s : float
        Standard deviation of the Gaussian kernel.
        
    Returns
    -------
    g : ndarray
        1D Gaussian kernel.
    dg : ndarray
        Derivative of the Gaussian kernel.
    x : ndarray
        x values where the Gaussian is computed.

    '''
    t = s**2

    r = np.ceil(4 * s)  #radio del kernel
    x = np.arange(-r, r + 1).reshape(-1, 1) # paso b crea el arrage con enteros 
    g = np.exp(-x**2 / (2 * t)) #formula gausianda de arriba 
    g = g/np.sum(g) #lo normaliza el kernel
    dg = -x * g / t 
    return g, dg, x

# Plot the Gaussian and its derivative
s = 4.5    # El parámetro σ (sigma) controla la anchura del kernel - mayor σ significa un kernel más ancho.
g, dg, x = get_gaussian_kernels(s)

fig, ax = plt.subplots()
ax.plot(x, g)
ax.plot(x, dg)
ax.set_title(f'Gaussian and its derivative, st.d. {s}')
plt.show()



# %% [markdown]
# ### Task 2 Verify separability 
# Un kernel gausaiano 2D puede separarse en dos kernels  1 d ortogonales
# 

# %%
# Read data
in_dir = './week1/'
im = imread(in_dir + 'fibres_xcth.png').astype(float)

#1D Gaussian kernel
# Get kernels
s = 4.5
g, dg, x = get_gaussian_kernels(s)
# Convolve image with two orthogonal 1D Gaussian kernels
im_g = convolve(convolve(im, g), g.T)   # convolucion



# 2D Gaussian kernel
g2D = g @ g.T
# Convolve image with 2D Gaussian kernel
im_2g = convolve(im, g2D)     #convolucion
  
# Show the results
fig, ax = plt.subplots(2, 2, figsize=(8, 8))    
ax[0][0].imshow(im_g, cmap='gray')
ax[0][0].set_title(f'Two orthogonal 1D kernels, st.d. {s}')
ax[0][1].imshow(im, cmap='gray')
ax[0][1].set_title(f'Original image, max value {im.max()}')
ax[1][0].imshow(im_2g, cmap='gray')
ax[1][0].set_title(f'One 2D kernel, st.d. {s}')
ax[1][1].imshow(im_g - im_2g, cmap='bwr', vmin=-1, vmax=1)
mad = np.abs(im_g - im_2g).mean()
ax[1][1].set_title(f'Difference between results, \nMean abs diff {mad:0.4g}')
fig.suptitle('Separability of Gaussians')
fig.tight_layout()
plt.show()




# %% [markdown]
# """ArithmeticErrorSeparabilidad: Un kernel 2D es separable si puede expresarse como producto externo de dos vectores (kernels 1D). Para los gaussianos, esto siempre es posible.
# 
# 
# La visualización muestra que la diferencia entre ambos métodos es mínima (MAD muy pequeño), verificando experimentalmente la separabilidad del kernel gaussiano.RetryJT
# """
# 

# %% [markdown]
# ### Task 3 Derivative of Gaussian vs. central difference

# %%
# Get kernels
s = 4.5
g, dg, x = get_gaussian_kernels(s)

# Primer método: Convolución directa con la derivada del gaussiano: 

# Método 1 (im_gx): Aplica directamente un kernel que es la derivada del gaussiano a la imagen original. Es una operación única que combina suavizado y derivación.
# Convolve with derivative of Gaussian 
im_gx = convolve(im, dg) #usa dg: derivada del gaussiano


# Segundo método: Convolución con gaussiano seguida de derivada (diferencia central):
# Método 2 (im_dx): Realiza dos operaciones secuenciales: primero suaviza la imagen con un kernel gaussiano y luego calcula la derivada con un kernel de diferencia central [0.5, 0, -0.5].
# Kernel for central difference image
k = np.array([[0.5, 0, -0.5]]).T
# Gaussian and central difference 
im_dx = convolve(convolve(im, g), k) #usa solo la g



# Show the results
fig, ax = plt.subplots(2, 2, figsize=(8, 8))    
ax[0][0].imshow(im_gx, cmap='bwr')
ax[0][0].set_title(f'Gaussian derivative, st.d. {s}')
ax[0][1].imshow(im, cmap='gray')
ax[0][1].set_title(f'Original image, max value {im.max()}')
ax[1][0].imshow(im_dx, cmap='bwr')
ax[1][0].set_title(f'Gaussian (st.d. {s}) and difference')
ax[1][1].imshow(im_gx - im_dx, cmap='bwr')
mad = np.abs(im_gx - im_dx).mean()
ax[1][1].set_title(f'Difference between results\nMean abs diff {mad:0.4g}')
fig.suptitle('Derivative og Gaussians vs. central difference')
fig.tight_layout()
plt.show()

# %% [markdown]
# ### Task 4 Large Gaussian compared to several small Gaussians

# %%

# (σ = √t)

# Small Gaussian
ts = 2
g, dg, x = get_gaussian_kernels(np.sqrt(ts))
im_gs = im.copy()
N = 10
for i in range(N):
    im_gs = convolve(convolve(im_gs, g), g.T)

# Large Gaussian
tl = 20
gl, dgl, xl = get_gaussian_kernels(np.sqrt(tl))  
im_gl = convolve(convolve(im, gl), gl.T)

# Show the results
fig, ax = plt.subplots(2, 2, figsize=(8, 8))    
ax[0][0].imshow(im_gs, cmap='gray')
ax[0][0].set_title(f'{N} convolutions with Gaussian t={ts}')
ax[0][1].imshow(im, cmap='gray')
ax[0][1].set_title(f'Original image, max value {im.max()}')
ax[1][0].imshow(im_gl, cmap='gray')
ax[1][0].set_title(f'One convolution with Gaussian t={tl}')
ax[1][1].imshow(im_gs - im_gl, cmap='bwr')
mad = np.abs(im_gs - im_gl).mean()
ax[1][1].set_title(f'Difference between results\nMean abs diff {mad:0.4g}')
fig.suptitle('Many small Gaussians vs. one large Gaussian')
fig.tight_layout()
plt.show()

# %% [markdown]
# 

# %% [markdown]
# ### Task 5 Gaussian derivatives combined with smoothing

# %%
# Small Gaussian derivative and Gaussian smoothing
ts = 10
g, dg, x = get_gaussian_kernels(np.sqrt(ts))
im_dg_s = convolve(convolve(im, g), dg)

# Large Gaussian derivative
tl = 20
gl, dgl, xl = get_gaussian_kernels(np.sqrt(tl))
im_dg_l = convolve(im, dgl)

# Show the results
fig, ax = plt.subplots(2, 2, figsize=(8, 8))    
ax[0][0].imshow(im_dg_s, cmap='bwr')
ax[0][0].set_title(f'Gaussian t={ts} and Gaussian derivative t={ts}')
ax[0][1].imshow(im, cmap='gray')
ax[0][1].set_title(f'Original image, max value {im.max()}')
ax[1][0].imshow(im_dg_l, cmap='bwr')
ax[1][0].set_title(f'Gaussian derivative t={tl}')
ax[1][1].imshow(im_dg_s - im_dg_l, cmap='bwr')
mad = np.abs(im_dg_s - im_dg_l).mean()
ax[1][1].set_title(f'Difference between results\nMean abs diff {mad:0.4g}')
fig.suptitle('Gaussian followed by derivative')
fig.tight_layout()
plt.show()

# %% [markdown]
#  Este ejercicio verifica otra propiedad importante de los kernels gaussianos y sus derivadas, relacionada con la combinación de operaciones. Si el MAD es pequeño, confirma que la derivada de un gaussiano grande puede descomponerse en operaciones con gaussianos más pequeños


