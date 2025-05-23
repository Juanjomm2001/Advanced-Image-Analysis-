# %% [markdown]
# # Curve smoothing exercise

# %%
import numpy as np
import scipy.linalg
import matplotlib.pyplot as plt

# Read data
in_dir = './week1/curves/'
X = np.loadtxt(in_dir + 'hand_noisy.txt')

# %% [markdown]
# ### Task 1
# Explicit (forward) smooting. ![image.png](attachment:image.png) alisado segun 1.10
# 

# %%
# Make circulant smoothing matrix
l = np.zeros(X.shape[0])
l[[-1, 0, 1]] = [1, -2, 1]
L = scipy.linalg.circulant(l)

# Smoothing parameter
lmb = 0.5

# Smooth the curve
X1 = (np.eye(X.shape[0]) + lmb * L) @ X

# Plot the results - use indexing to plot closed curve
idx = np.arange(X.shape[0] + 1)
idx[-1] = 0
fig, ax = plt.subplots()
ax.plot(X[idx, 0], X[idx, 1], alpha=0.5)
ax.plot(X1[idx, 0], X1[idx, 1])
ax.set_aspect('equal')
ax.set_title(f'Smoothing once with lambda = {lmb}')
plt.show()

# %% [markdown]
# Comportamiento según el enunciado:
# 
# *   Con λ=0.5, una iteración coloca cada punto exactamente en el promedio de sus vecinos
# Al aumentar λ, el suavizado se vuelve más agresivo y pueden aparecer inestabilidades 
#     'si λ>1
# Con λ pequeño (0.01), se necesitan más iteraciones para lograr un efecto visible
# 
# 
# El código muestra tanto la curva original (transparente) como la curva suavizada después de 100 iteraciones con λ=0.01, permitiendo comparar visualmente el resultado del suavizado.

# %%
# Smoothing parameters
n_iter = 100
lmb = 0.01  

# Smooth the curve
Xn = X.copy()
for i in range(n_iter):
    Xn = (np.eye(X.shape[0]) + lmb * L) @ Xn

# Show the results
fig, ax = plt.subplots()
ax.plot(X[idx, 0], X[idx, 1], alpha=0.5) 
ax.plot(Xn[idx, 0], Xn[idx, 1])
ax.set_aspect('equal')
ax.set_title(f'Smoothing with lambda {lmb} for {n_iter} iterations')
plt.show()

        # np.eye(X.shape[0]) es la matriz identidad
        # lmb es el parámetro lambda (λ) que controla la intensidad del suavizado
        # L es la matriz Laplaciana 
        # @ representa multiplicación matricial

# %% [markdown]
# Let's visualize every iteration using `ipwidgets`.

# %%
from ipywidgets import interact

n_iter = 100
lmb = 0.5  

X_iters = [X]
for i in range(n_iter):
    X_iters.append((np.eye(X.shape[0]) + lmb * L) @ X_iters[-1])

def show_iter(i):
    plt.plot(X[idx, 0], X[idx, 1], alpha=0.5) 
    plt.plot(X_iters[i][idx, 0], X_iters[i][idx, 1])
    plt.title(f'Smoothing with lambda {lmb} iter {i}')
    plt.gca().set_aspect('equal')
    plt.show()

interact(show_iter, i=(0, n_iter, 1));

# %% [markdown]
# ### Task 2 Implicit smoothing
# ![image.png](attachment:image.png)
# With implicit smoothing, we can use a large smoothing parameter.

# %%
# Smoothing parameter and smoothing
lmb = 3.5 
X_implicit = np.linalg.inv(np.eye(X.shape[0]) - lmb * L) @ X

# Show the results
fig, ax = plt.subplots()
ax.set_aspect('equal')
ax.plot(X[idx,0], X[idx,1], alpha=0.5)
ax.plot(X_implicit[idx,0], X_implicit[idx,1])
ax.set_title(f'Implicit smoothing lambda {lmb}')
plt.show()

# %% [markdown]
# Do you need an iterative approach
# of this smoothing?  No, esta es la ventaja principal del método implícito. Calcula la solución directamente resolviendo un sistema lineal, sin necesidad de iteraciones.
# 
# - Valores más altos (como 3.5 usado aquí) producen un suavizado más intenso
# 
# 
# 
# 

# %% [markdown]
# ### Task 3 Extended kernel
# 
# En lugar de usar solo la matriz Laplaciana (L), se utilizan dos matrices diferentes: A y B
# La ecuación se modifica para usar αA + βB en vez de solo λL

# %%
# Circulant smoothing matrices
a = np.zeros(X.shape[0]) # Array for first order derivative
a[[-1, 0, 1]] = [1, -2, 1]
A = scipy.linalg.circulant(a) # Circulant matrix for first order derivative
b = np.zeros(X.shape[0]) # Array for second order derivative
b[[-2, -1, 0, 1, 2]] = [-1, 4, -6, 4, -1]
B = scipy.linalg.circulant(b) # Circulant matrix for second order derivative

# Smoothing parameters
alpha = 0.1
beta = 1

# Smooth the curve
Xn = X.copy()
Xn = np.linalg.inv(np.eye(X.shape[0]) - alpha*A - beta*B)@Xn

# Show the results
fig, ax = plt.subplots()
ax.set_aspect('equal')
ax.plot(X[idx,0], X[idx,1])
ax.plot(Xn[idx,0], Xn[idx,1])
ax.set_title(f'Extended kernel with alpha = {alpha} and beta = {beta}')
plt.show()

# %% [markdown]
# Parámetros α y β:
# 
# - α controla la influencia de la primera derivada (suavizado de la curvatura)
# -   β controla la influencia de la segunda derivada (suavizado de cambios en la curvatura)
# - Con β grande y α pequeño (como en el ejemplo: α=0.1, β=1), se enfatiza el suavizado de las variaciones de curvatura
# Esto tiende a preservar mejor la forma general de la curva mientras elimina pequeñas irregularidades

# %% [markdown]
# ### Task 4 Making a function for smoothing matrix

# %%
def regularization_matrix(N, alpha, beta):
    '''Returns circulant N x N matrix for imposing elasticity and rigidity to snakes.
     
    Parameters
    ----------
    N : int
        Number of points in the curve.

    alpha : float
        Weight for elasticity.

    beta : float
        Weight for rigidity.

    Returns
    -------
    ndarray
        Circulant N x N smoothing matrix.
    
    '''
    s = np.zeros(N)
    s[[-2, -1, 0, 1, 2]] = (alpha * np.array([0, 1, -2, 1, 0]) + 
                    beta * np.array([-1, 4, -6, 4, -1]))
    S = scipy.linalg.circulant(s)  
    return scipy.linalg.inv(np.eye(N) - S)

# Smoothing parameters and smoothing
alpha = 0.1
beta = 1
X_smoothed = regularization_matrix(X.shape[0], alpha, beta) @ X

# Show the results
fig, ax = plt.subplots()
ax.set_aspect('equal')
ax.plot(X[idx, 0], X[idx, 1])
ax.plot(X_smoothed[idx, 0], X_smoothed[idx, 1])
ax.set_title(f'Smoothed with alpha = {alpha} and beta = {beta}')
plt.show()

# %% [markdown]
# ### Extra: interactive display

# %%
from ipywidgets import interact

X_smoothed = {}
for a in np.round(np.arange(0, 10.1, 0.25), 2): #  round to avoid floating point errors
    for b in np.round(np.arange(0, 10.1, 0.25), 2):
        X_smoothed[a, b] = regularization_matrix(X.shape[0], a, b) @ X

def show_result(a, b):
    plt.plot(X[idx, 0], X[idx, 1], alpha=0.5) 
    plt.plot(X_smoothed[a, b][idx, 0], X_smoothed[a, b][idx, 1])
    plt.title(f'Smoothing with alpha {a} beta {b}')
    plt.gca().set_aspect('equal')
    plt.show()

interact(show_result, a=(0, 10, 0.25), b=(0, 10, 0.25));

# %%



