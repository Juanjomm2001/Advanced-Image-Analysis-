#%%
import numpy as np
import matplotlib.pyplot as plt
import sklearn.datasets

#%% importing mlp from week 8
import os
import sys

mlp_path = os.path.dirname('/Users/VAND/Documents/TEACHING/02506/exercises/Week08/mlp_basic.py')
sys.path.append(mlp_path)

import mlp_basic as mlp

#%%


digits = sklearn.datasets.load_digits()
images = digits['images']
targets = digits['target']

X = images.reshape((images.shape[0], -1)).T
h = X.max()/2
X = (X - h)/h
I = np.eye(10, dtype=bool)
T = I[targets].T

def disp(i):
    '''Helping function whens showing images'''
    return (i.reshape(8, 8) + 1) / 2

nr_images = images.shape[0]
nr_show = 8
rg = np.random.default_rng()
random_subset = sorted(rg.choice(range(nr_images), nr_show, replace=False))

fig, ax = plt.subplots(1, nr_show)
for s, a in zip(random_subset, ax):
    a.imshow(disp(X[:, s]))
    a.set_title(f'Im. {s}\nTarget {np.argmax(T[:, s])}')
plt.show()


#%% Make testing and training set
train_percentage = 0.5
permuted = rg.permutation(range(nr_images))
c = int(nr_images*train_percentage)

train = sorted(permuted[:c])
test = sorted(permuted[c:])


X_train= X[:, train]
T_train= T[:, train]

X_test = X[:, test]
T_test = T[:, test]


#%% Initialization.
mlp_size = (X_train.shape[0], 58, 22, T_train.shape[0])
W = mlp.initialize(mlp_size)

# Training parameters.
nr_epoch = 10
batchsize = 5
eta = 0.01
losses = []
losses_test = []

#%%
# Training.

nr_points = X_train.shape[1]

for e in range(nr_epoch):

    random_order = rg.permutation(range(nr_points))
    epoch_loss = 0

    for k in range(0, nr_points, batchsize):
        
        batch = random_order[k:k+batchsize]  
        X_batch = X_train[:, batch]
        T_batch = T_train[:, batch]

        W, loss = mlp.backward(X_batch, T_batch, W, eta)
        epoch_loss += loss

    losses.append(epoch_loss/nr_points)
    
    Y_test = mlp.predict(X_test, W)
    loss_test = Y_test[T_test]  # boolean indexing instead of (t * np.log(x)).sum(axis=0)
    loss_test = - np.log(loss_test)
    losses_test.append(loss_test.mean())

    
    print(f'\rEpoch {e}, loss {epoch_loss}', end=' ' * 20)


#%%

Y_train = mlp.predict(X_train, W)
predicted_train = np.argmax(Y_train, axis=0)
target_train= np.argmax(T_train, axis=0)
accuracy_train = (predicted_train==target_train).sum()/target_train.size

Y_test = mlp.predict(X_test, W)
predicted_test = np.argmax(Y_test, axis=0)
target_test = np.argmax(T_test, axis=0)
accuracy_test = (predicted_test==target_test).sum()/target_test.size

#%%

def perturbe(x, scale=0.1):
    return x + rg.normal(scale=scale, size=x.shape)

# Visualization.
fig, ax = plt.subplots(1, 3)
ax[0].plot(losses, label='train')
ax[0].plot(losses_test, label='test')

ax[0].set_xlabel('Epoch')
ax[0].set_ylabel('Loss') 


ax[1].scatter(perturbe(target_train), perturbe(predicted_train), alpha=0.5, s=15)
ax[1].set_aspect('equal', 'box')
ax[1].set_xlabel('Target')
ax[1].set_ylabel('Predicted')
ax[1].set_title(f'Train: {int(accuracy_train*100):d}%')

ax[2].scatter(perturbe(target_test), perturbe(predicted_test), alpha=0.5, s=15)
ax[2].set_aspect('equal', 'box')
ax[2].set_xlabel('Target')
ax[2].set_ylabel('Predicted')
ax[2].set_title(f'Test: {int(accuracy_test*100):d}%')
plt.show()

#%% Check where it goes wrong
misses = np.where(target_test != predicted_test)[0]
nr_show = min(nr_show, len(misses))

if nr_show>0:
    fig, ax = plt.subplots(1, nr_show)
    for i, a in zip(misses, ax):
        a.imshow(disp(X_test[:, i]))
        tr = np.argmax(T_test[:, i])
        pr = np.argmax(Y_test[:, i])
        a.set_title(f'Predicted {pr}\nTarget {tr}')
    plt.show()



# %%
