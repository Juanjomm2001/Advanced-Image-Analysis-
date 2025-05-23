#%%
import numpy as np
import matplotlib.pyplot as plt
import make_data
%matplotlib tk



# %%
example_nr = 1
n_pts = 1000
noise = 3
X, T, x_grid, dim = make_data.make_data(example_nr, n_pts, noise)
mu = X.mean(axis=1, keepdims=True)
std = X.std(axis=1, keepdims=True)
print(mu)
print(std)
#%%

X_c = (X-mu)/std

fig, ax = plt.subplots()
ax.plot(X_c[0,T[0]], X_c[1,T[0]], '.r', alpha=0.3)
ax.plot(X_c[0,T[1]], X_c[1,T[1]], '.g', alpha=0.3)
# ax.set_xlim(0, 100)
# ax.set_ylim(0, 100)
ax.set_box_aspect(1)

# %%
n = 5
x = X_c[:,0:n]
x.shape
w = np.sqrt(1/3)
W = []
W.append(w*np.random.randn(3,3))
W.append(w*np.random.randn(4,2))

# %%

x = np.vstack((x, np.ones((1,n))))
x.shape
# %%

z = W[0].T@x
h = np.maximum(0, z)
y_hat = W[1].T@np.vstack((h, np.ones((1,n))))
y = np.exp(y_hat)/(np.exp(y_hat).sum(axis=0))

print(y)
# %%
dims = [X.shape[0], 3, 2]
def init(dims):
    W = []
    # do something here
    return W

def forward(X, W):
    h = None
    y = None
    # do something here
    return y, h

def backward(X, T, W, lr=0.001):
    y, h = forward(X, W)
    # do something here
    return W

