#%%

import numpy as np

#%%
folder = '/Users/VAND/Documents/TEACHING/02506/data/week9_MNIST_data/'
d = np.load(folder + 'MNIST_target_train.npy')


#%%


folder = '/Users/VAND/Documents/TEACHING/02506/data/week9_CIFAR-10_data/'
d = np.load(folder + 'CIFAR10_target_train.npy')


# %%
file = '/Users/VAND/Documents/TEACHING/02506/data/cifar-10-batches-py/data_batch_1'

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

d = unpickle(file)

d[b'data'].shape