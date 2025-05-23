#%%

import numpy as np

rg = np.random.default_rng()

def initialize(mlp_size):
    
    W = []
    for l in range(len(mlp_size) - 1):
        size=(mlp_size[l] + 1, mlp_size[l + 1])
        W.append(rg.normal(scale=np.sqrt(2/size[0]), size=size))
    return W


def predict(x, W):
    '''Returns y without saving hidden layers.'''
    
    c = x
    for l in range(len(W) - 1):
        c = W[l].T @ np.vstack((c, np.ones(c.shape[1])))
        c = np.maximum(c, 0)
    c = W[-1].T @ np.vstack((c, np.ones(c.shape[1])))

    c = np.exp(c)
    c = np.clip(c, 1e-15, 1e15)  # to avoid division by 0 a
    c *= (1 / c.sum(axis=0))

    return c

def forward(x, W):
    '''Returns hidden layers with yhat as the last element.'''

    h = []
    c = x

    for l in range(len(W) - 1):
        c = W[l].T @ np.vstack((c, np.ones(c.shape[1])))
        c = np.maximum(c, 0)
        h.append(c)

    c = W[-1].T @ np.vstack((c, np.ones(c.shape[1])))
    h.append(c)
    return h


def backward(x, t, W, eta):
    '''Returns updated W and sum of losses.'''

    h = forward(x, W)
    m = x.shape[1]
    
    # Softmax.
    y = np.exp(h[-1])
    y = np.clip(y, 1e-15, 1e15)  # to avoid division by 0 and log(0)
    y *= (1 / y.sum(axis=0))
    
    # Loss.
    loss = y[t]  # boolean indexing instead of (t * np.log(x)).sum(axis=0)
    loss = - np.log(loss)
    loss = loss.sum() 
    
    # Delta for last layer.
    delta = y - t  

    # Move backward.
    for l in range(len(W) - 1, 0, -1):
        
        Q = np.vstack((h[l-1], np.ones(h[l-1].shape[1]))) @ delta.T
        delta = W[l][:-1, :] @ delta
        delta *= h[l-1]>0   #  Adding activation part.
        W[l] -= (eta/m) * Q  #  Update.
        
    # First layer.  
    Q = np.vstack((x, np.ones(x.shape[1]))) @ delta.T
    W[0] -= eta * Q

    return W, loss


def train(X, T, W, nr_epoch, eta, batchsize=1, losses=[]):

    nr_points = X.shape[1]

    for e in range(nr_epoch):
    
        random_order = rg.permutation(range(nr_points))
        epoch_loss = 0

        for k in range(0, nr_points, batchsize):
            
            batch = random_order[k:k+batchsize]  
            X_batch = X[:, batch]
            T_batch = T[:, batch]

            W, loss = backward(X_batch, T_batch, W, eta)
            epoch_loss += loss
            
        losses.append(epoch_loss)
        
        print(f'\rEpoch {e}, loss {epoch_loss}', end=' ' * 20)

    return W, losses


#%% 
# SCRIPT STARTS HERE
#%% Test of the data generation
if __name__ == "__main__":

    from make_data import make_data
    import matplotlib.pyplot as plt

    #  Data
    X, T, grid_X, grid_dim = make_data(2, 200, noise = 0.9)
    nr_points = X.shape[1]
    X_c = (X - 50)/50
    grid_X = (grid_X - 50)/50 

    # Initialization.
    mlp_size = (2, 3, 2)
    W = initialize(mlp_size)

    # Training parameters.
    nr_epoch = 100
    batchsize = 1
    eta = 0.05
    losses = []

    #%%
    # Training.
    W, losses = train(X_c, T, W, nr_epoch, eta, batchsize=batchsize, losses=losses)

    grid_Y = predict(grid_X, W)
    prob0 = grid_Y[0].reshape(grid_dim)

    #%%
    # Visualization.
    fig, ax = plt.subplots(1, 2)
    ax[0].plot(losses)
    ax[0].set_xlabel('Epoch')
    ax[0].set_ylabel('Loss')
    ax[1].imshow(prob0, cmap=plt.cm.bwr, vmin=0.45, vmax=0.55)

    ax[1].scatter(X[0][T[0]], X[1][T[0]], c='m', alpha=0.5, s=15)
    ax[1].scatter(X[0][T[1]], X[1][T[1]], c='g', alpha=0.5, s=15)
    ax[1].set_aspect('equal', 'box')
    plt.show()


# %%
