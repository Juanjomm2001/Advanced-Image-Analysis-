#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 24 14:33:58 2021

@author: abda
"""

#%%
import numpy as np
import matplotlib.pyplot as plt
import make_data

#%%
n = 500
example_nr = 1
noise = 1

X, T, x, dim = make_data.make_data(example_nr, n, noise)
fig, ax = plt.subplots(1,1)
ax.scatter(X[0:n,0],X[0:n,1],c = 'red', alpha = 0.3, s = 15)
ax.scatter(X[n:2*n,0],X[n:2*n,1],c = 'green', alpha = 0.3, s = 15)
ax.set_aspect('equal', 'box')
plt.title('training')
fig.show

#%%

Xo = x.reshape((100,100,2))
plt.imshow(Xo[:,:,1])


#%%

m = np.mean(X,axis = 0)
s = np.std(X,axis = 0)

Xc = (X - m)/s
xc = (x - m)/s

#%%


n_hidden = 10
W = []
W.append(np.random.randn(3,n_hidden)*np.sqrt(2/3))
W.append(np.random.randn(n_hidden+1,2)*np.sqrt(2/4))

def forwardsimple(xb, W):
    n_pts = xb.shape[0]
    z = np.c_[xb, np.ones(n_pts)]@W[0]
    h = np.maximum(z, 0)
    yh = np.c_[h, np.ones(n_pts)]@W[1]
    y = np.exp(yh)/np.sum(np.exp(yh),axis=1,keepdims=True)
    return y, h

y, h = forwardsimple(xc, W)


Y = y.reshape((100,100,2))
plt.imshow(Y[:,:,1])






















# %%
