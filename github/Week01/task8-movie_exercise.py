# %% [markdown]
# # Movie exercise
# 
# Estimate growth of listeria bacteria from the movie.

# %%
from imageio import get_reader
import matplotlib.pyplot as plt
from skimage.color import rgb2gray
from scipy.ndimage import gaussian_gradient_magnitude as ggm

# %%
# Read data
in_dir = './week1/'
vid = get_reader(in_dir + 'listeria_movie.mp4',  'ffmpeg')

frames = []
for v in vid.iter_data():
    frames.append(rgb2gray(v))  #Frames

fig, ax = plt.subplots(1, 5)
for i in range(5):
    k = int(i * (len(frames) - 1) / 4)
    ax[i].imshow(frames[k], cmap='gray')
    ax[i].set_title(f'Frame {k}') 
plt.show()

# %%
# Find parameters which give a good result 
sigma = 2
threshold = 0.003

fig, ax = plt.subplots(2, 5)
for i in range(5):
    k = int(i * (len(frames) - 1) / 4)
    im_l = ggm(frames[k], sigma)  # es una función que calcula el gradiente gaussiano (Gaussian Gradient Magnitude)
    ax[0, i].imshow(im_l)
    ax[0, i].set_title(f'Frame {k}') 
    ax[1, i].imshow(im_l > threshold)
plt.show()

# %%
# Count bacteria for each frame in the movie
bacteria = [(ggm(f, sigma) > threshold).mean() for f in frames]
# Show the results
fig, ax = plt.subplots()
ax.plot(bacteria)
ax.set_xlabel('Frame number')
ax.set_ylabel('Bacteria count')
ax.set_title('Bacteria growth - sigmoid shape')
plt.show()

# %% [markdown]
# La gráfica resultante muestra una forma sigmoidea (en forma de "S"), característica del crecimiento bacteriano, que corresponde al modelo logístico de crecimiento poblacional:


