# %% [markdown]
# # Tracking amoeba

# %%
import imageio
import numpy as np
import matplotlib.pyplot as plt
import simple_snake_new as sis
%matplotlib inline  

# %% [markdown]
# ## Investigate data

# %%
filename = './week6/crawling_amoeba.mov'
vid = imageio.get_reader(filename)
movie = np.array([im for im in vid.iter_data()], dtype=float)/255
movie = movie.mean(axis=3)

samples = [0, 100, 250]
fig, ax = plt.subplots(1, len(samples), figsize=(10, 3))
for s, a in zip(samples, ax):
    a.imshow(movie[s],cmap='gray')
    a.set_title(f'Frame {s}')
plt.show()

# %% [markdown]
# ## Initialize snake

# %%
#%% settings
nr_points = 100    # Número de puntos en la snake
step_size = 10     # Tamaño del paso de evolución
alpha = 0.02       # Regularización: continuidad (elasticidad)
beta = 0.02        # Regularización: suavidad (rigidez)
center = (120, 200) # Centro de la snake inicial (y, x)
radius = 40        # Radio de la snake circular inicial
#%% initialization
snake = sis.make_circular_snake(nr_points, center, radius)
B = sis.regularization_matrix(nr_points, alpha, beta)
frame = movie[0]
fig, ax = plt.subplots()
ax.imshow(frame, cmap='gray')
closed = np.hstack([np.arange(nr_points), 0])  # Indices of the closed curve
ax.plot(snake[closed, 1], snake[closed, 0], 'b-')
ax.set_title('Initialization')
plt.show()

# %% [markdown]
# ## Segment amoeba in the first frame

# %%
# Este código implementa la evolución iterativa de la snake en un solo frame (paso 9 del enunciado)
fig, ax = plt.subplots()
for i in range(120):    
    snake = sis.evolve_snake(snake, frame, B, step_size)    
    ax.clear()
    ax.imshow(frame, cmap='gray')
    ax.plot(snake[closed, 1], snake[closed, 0], 'b-')
    ax.set_title(f'Iter {i}')
    plt.pause(0.001)
plt.show()

# %% [markdown]
# ## Track amoeba

# %%
fig, ax = plt.subplots()
for i in range(250):
    frame = movie[i] 
    snake = sis.evolve_snake(snake, frame, B, step_size)    
    ax.clear()
    ax.imshow(frame, cmap='gray')
    ax.plot(snake[closed, 1], snake[closed, 0], 'b-')
    ax.set_title(f'tracking, frame {i}')
    plt.pause(0.001)
plt.show()



