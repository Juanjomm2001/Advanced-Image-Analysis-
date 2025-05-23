#%%
# Imports and defs

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors
import PIL.Image
import numpy as np

import os
import sys
mlp_path = os.path.dirname('/Users/VAND/Documents/TEACHING/02506/exercises/Week08/mlp_basic.py')
sys.path.append(mlp_path)
import mlp_basic as mlp


def disp(i):
    '''Helping function whens showing images'''
    return (i.reshape(imsize) + 1) / 2


def perturbe(x, scale=0.1):
    return x + rg.normal(scale=scale, size=x.shape)


def show_confusion_scatter(ax, target, predicted):
    ax.scatter(perturbe(target), perturbe(predicted), alpha=0.5, s=15)
    ax.set_xlim(-0.5, 11.5)
    ax.set_ylim(-0.5, 11.5)
    ax.set_aspect('equal', 'box')
    ax.set_xlabel('Target')
    ax.set_ylabel('Predicted')
    

def show_confusion_matrix(ax, target, predicted):
    nr_classes = target.max() + 1
    nr_points = target.size
    edges = np.arange(nr_classes + 1) - 0.5
    cm = np.histogram2d(predicted, target, [edges, edges])[0]
    i = ax.imshow(cm + 1, cmap=plt.cm.plasma, norm=matplotlib.colors.LogNorm())  # log color
    ax.set_xlim(edges[0], edges[-1])
    ax.set_ylim(edges[0], edges[-1])
    ax.set_aspect('equal', 'box')
    ax.set_xlabel('Target')
    ax.set_ylabel('Predicted')


def evaluate_result(W, X_train, T_train, X_validate, T_validate, losses_running, losses_validate, losses_batch):

    Y_train = mlp.predict(X_train, W)
    predicted_train = np.argmax(Y_train, axis=0)
    target_train= np.argmax(T_train, axis=0)
    accuracy_train = (predicted_train==target_train).sum()/target_train.size

    Y_validate = mlp.predict(X_validate, W)
    loss_validate = Y_validate[T_validate]  # boolean indexing instead of (t * np.log(x)).sum(axis=0)
    loss_validate = - np.log(loss_validate)
    predicted_validate = np.argmax(Y_validate, axis=0)
    target_validate = np.argmax(T_validate, axis=0)
    accuracy_validate = (predicted_validate==target_validate).sum()/target_validate.size

    losses_validate.append(loss_validate.mean())

    # Visualization.
    fig = plt.figure()
    gs = fig.add_gridspec(2, 2)
    ax1 = fig.add_subplot(gs[1, 0])
    ax2 = fig.add_subplot(gs[1, 1])
    ax = fig.add_subplot(gs[0, :])

    timestamp = np.arange(len(losses_running))
    ax.plot(losses_batch[0], losses_batch[1], lw=0.2, alpha=0.5, label='Batches')
    ax.plot(timestamp + 0.5, losses_running, lw=0.5, label='Train (running)')
    ax.plot(timestamp + 1, losses_validate, lw=0.5, label='Validate')
    ax.set_ylim(0, losses_running[0])
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss') 
    ax.legend()

    show_confusion_matrix(ax1, target_train, predicted_train)
    ax1.set_title(f'Train: {int(accuracy_train*100):d}%')
    show_confusion_matrix(ax2, target_validate, predicted_validate)
    ax2.set_title(f'Validate: {int(accuracy_validate*100):d}%')

    fig.suptitle(f'Epoch {len(losses_running)}')
    plt.tight_layout()
    plt.show()

    return losses_validate



#%% 
# Set-up data

path = '/Users/VAND/Documents/TEACHING/02506/data/bugNIST2D/'  # path to unzipped data directory
train_filenames = sorted(os.listdir(path + 'train')) 
train_targets = np.loadtxt(path + 'train_targets.txt', dtype=int) 

X_train = np.stack([np.array(PIL.Image.open(path + 'train/' + filename)) for filename in train_filenames], axis=-1)
imsize = X_train.shape[:2]
nr_images = X_train.shape[3]
X_train = 2/255 * X_train.reshape(-1, nr_images).astype('float') - 1

I = np.eye(12, dtype=bool)
T_train = I[train_targets].T

split = 20000
X_validate = X_train[:, split:]
T_validate = T_train[:, split:]

X_train = X_train[:, :split]
T_train = T_train[:, :split]


#%% 
# Initialize MLP.

nr_show = 8
rg = np.random.default_rng()

mlp_size = (X_train.shape[0], 256, 512, 256, T_train.shape[0])
W = mlp.initialize(mlp_size)
losses_running = []
losses_validate = []
losses_batch = [[], []]

#%% 
# Set training parameters.
nr_epoch = 200
batchsize = 20
eta = 0.001

#%%
# Train.

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
        losses_batch[0].append(e + k/nr_points)
        losses_batch[1].append(loss/X_batch.shape[1])

    losses_running.append(epoch_loss/nr_points)
    losses_validate = evaluate_result(W, X_train, T_train, X_validate, T_validate, losses_running, losses_validate, losses_batch)
    
    #print(f'\rEpoch {e}, loss {epoch_loss}', end=' ' * 20)


#%% 
# Testing

test_filenames = sorted(os.listdir(path + 'test')) 
test_targets = np.loadtxt(path + 'test_targets.txt', dtype=int) 
X_test = np.stack([np.array(PIL.Image.open(path + 'test/' + filename)) for filename in test_filenames], axis=-1)
X_test = 2/255 * X_test.reshape(-1, X_test.shape[-1]).astype('float') - 1
T_test = I[test_targets].T


Y_test = mlp.predict(X_test, W)
loss_test = Y_test[T_test]  # boolean indexing instead of (t * np.log(x)).sum(axis=0)
loss_test = - np.log(loss_test)
predicted_test = np.argmax(Y_test, axis=0)
target_test = np.argmax(T_test, axis=0)
accuracy_test = (predicted_test==target_test).sum()/target_test.size

fig, ax = plt.subplots()
show_confusion_matrix(ax, target_test, predicted_test)
ax.set_title(f'Test: {int(accuracy_test*100):d}%')


# %%
I = np.array([1, 2, 4, 9, 6, 5, 3, 8, 11, 7, 10, 0]) ## which place it should move to

fig, ax = plt.subplots()
show_confusion_matrix(ax, I[target_test], I[predicted_test])
ax.set_title(f'Test: {int(accuracy_test*100):d}%')

# %%
