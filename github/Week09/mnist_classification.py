#%%
import gzip
import shutil
import os
import wget


# This will get and unpack mnist files in the specified folder.
# Alternatively, download, unpack and place in folder manually. 
# http://yann.lecun.com/exdb/mnist/train-images-id3-ubyte.gz
# http://yann.lecun.com/exdb/mnist/train-labels-id1-ubyte.gz
# http://yann.lecun.com/exdb/mnist/t10k-images-id3-ubyte.gz
# http://yann.lecun.com/exdb/mnist/t10k-labels-id1-ubyte.gz

mnist_folder = '/Users/VAND/Documents/TEACHING/02506/data/mnist_data'

if not os.path.isdir(mnist_folder):
    print('Getting data...')
    os.mkdir(mnist_folder)
    for f in ['train-images-idx3-ubyte', 'train-labels-idx1-ubyte', 
              't10k-images-idx3-ubyte', 't10k-labels-idx1-ubyte']:
        url = 'http://yann.lecun.com/exdb/mnist/' + f + '.gz'
        temp = wget.download(url) 
        with gzip.open(temp, 'rb') as f_in:  
            with open(os.path.join(mnist_folder, f), 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
        os.remove(temp)
else:
     print('Has data.')

def disp(i):
    '''Helping function whens showing images'''
    return (i.reshape(28, 28) + 1) / 2

#%%
import mnist
import numpy as np

mndata = mnist.MNIST(mnist_folder)
X_train, T_train = mndata.load_training()
X_test, T_test = mndata.load_testing()

X_train = (2/255) * np.array(X_train, dtype=float).T - 1 
X_test = (2/255) * np.array(X_test, dtype=float).T  - 1

I = np.eye(10, dtype=bool)
T_train = I[T_train].T
T_test = I[T_test].T


#%%
#%% importing mlp from week 8
import os
import sys

mlp_path = os.path.dirname('/Users/VAND/Documents/TEACHING/02506/exercises/Week08/mlp_basic.py')
sys.path.append(mlp_path)

import mlp_basic as mlp


#%%
X_train -= X_train.mean(axis=0)
X_train /= X_train.std(axis=0)

X_test -= X_test.mean(axis=0)
X_test /= X_test.std(axis=0)

#%%
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

rg = np.random.default_rng()

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
import matplotlib.pyplot as plt

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

nr_show= 8




misses = np.where(target_test != predicted_test)[0]
nr_show = min(nr_show, len(misses))

if nr_show>0:
    fig, ax = plt.subplots(1, nr_show)
    for i, a in zip(misses, ax):
        a.imshow(disp(X_test[:, i]))
        tr = np.argmax(T_test[:, i])
        pr = np.argmax(Y_test[:, i])
        a.set_title(f'{pr} ({tr})')
    plt.show()

# %%
