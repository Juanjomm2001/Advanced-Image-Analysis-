#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Anders B. Dahl, abda@dtu.dk, March 2021 
"""

#%% Generate and display data
import make_data
import numpy as np
import matplotlib.pyplot as plt

n = 100
example_nr = 2
noise = 1.75

X, T, x, dim = make_data.make_data(example_nr, n, noise)

# Standardize
m = np.mean(X,axis=0)
s = np.std(X,axis=0)
Xc = (X-m)/s
xc = (x-m)/s


fig, ax = plt.subplots(1)
ax.plot(Xc[:n,0],Xc[:n,1],'r.',markersize=10,alpha=0.3)
ax.plot(Xc[n:,0],Xc[n:,1],'g.',markersize=10,alpha=0.3)
ax.set_aspect('equal')


#%% Forward simple model

# Function for simple forward pass
def simple_forward(x, W):
    z = np.c_[x,np.ones((x.shape[0]))]@W[0]
    h = np.maximum(z,0)
    yh = np.c_[h,np.ones((x.shape[0]))]@W[1]
    y = np.exp(yh)/np.sum(np.exp(yh),axis=1,keepdims=True)
    return y, h

# Function for simple backpropagation
def simple_backward(x, W, t, learning_rate=0.1):
    y, h = simple_forward(x,W)
    L = -np.sum(t*np.log(y + 10e-10))/x.shape[0]
    # print(L)
    d1 = y - t
    q1 = np.c_[h,np.ones((x.shape[0]))].T@d1/y.shape[0]
    d0 = (h>0)*(d1@W[1].T)[:,:-1]
    q0 = np.c_[x,np.ones((x.shape[0]))].T@d0/y.shape[0]
    W[0] -= learning_rate*q0
    W[1] -= learning_rate*q1
    return W, L

# Function for simple weight initializaion
def simple_init_weights(n):
    W = []
    W.append(np.random.randn(3,n)*np.sqrt(2/3))
    W.append(np.random.randn(n+1,2)*np.sqrt(2/(n+1)))
    return W

W = simple_init_weights(3)

fig, ax = plt.subplots(1)
n_iter = 50
L = np.zeros((n_iter))
i_rng = np.arange(0,n_iter) 
for i in range(0,n_iter):
    W, L[i] = simple_backward(Xc,W,T,learning_rate = 0.5)
    ax.cla()
    ax.plot(i_rng,L,'k')
    ax.set_title('Loss')
    plt.pause(0.001)
    plt.show()
    
    
y = simple_forward(xc,W)[0]

# Display the result
Y = y.reshape((100,100,2))
fig,ax = plt.subplots(1)
ax.imshow(Y[:,:,1],cmap='pink')
ax.plot(X[:n,0],X[:n,1],'r.',markersize=10,alpha=0.3)
ax.plot(X[n:,0],X[n:,1],'g.',markersize=10,alpha=0.3)
ax.set_aspect('equal')

#%% Varying number of layers

nl = [x.shape[1], 50,50,50,50, 2]
def init_weights(nl):
    W = []
    for i in range(1,len(nl)):
        W.append(np.random.randn(nl[i-1]+1,nl[i])*np.sqrt(2/(nl[i-1]+1)))
    return W

W = init_weights(nl)

def forward(x, W):
    n = len(W)
    z = []
    h = []
    z.append(np.c_[x,np.ones((x.shape[0]))]@W[0])
    h.append(np.maximum(z[0],0))
    for i in range(1,n-1):
        z.append(np.c_[h[i-1],np.ones((x.shape[0]))]@W[i])
        h.append(np.maximum(z[i],0))
    
    yh = np.maximum(np.minimum(np.c_[h[-1],np.ones((x.shape[0]))]@W[-1],600),-600)
    y = np.exp(yh)/(np.sum(np.exp(yh),axis=1,keepdims=True))
    return y, h

def backward(x, W, t, learning_rate=0.01, show_learning=False):
    y,h = forward(x,W)
    L = -np.sum(t*np.log(y + 10e-7))/x.shape[0]
    if show_learning:
        print(L)
    n = len(W)
    d = []
    q = []
    d.append(y - t)
    for i in range(1,n):
        q.append((np.c_[h[n-i-1],np.ones((x.shape[0]))].T@d[i-1])/y.shape[0])
        d.append((h[n-i-1]>0)*(d[i-1]@W[n-i].T)[:,:-1])
    q.append((np.c_[x,np.ones((x.shape[0]))].T@d[-1])/y.shape[0])

    for i in range(0,n):
        W[i] -= learning_rate*q[n-i-1]
    return W, L



fig, ax = plt.subplots(1)
n_iter = 250
L = np.zeros((n_iter))
i_rng = np.arange(0,n_iter) 
for i in range(0,n_iter):
    W, L[i] = backward(Xc,W,T,learning_rate=0.1,show_learning=True)
    if ( i%10 == 0):
        ax.cla()
        ax.plot(i_rng,L,'k')
        ax.set_title('Loss')
        plt.pause(0.001)
        plt.show()


y = forward(xc,W)[0]

Y = y.reshape((100,100,2))
fig,ax = plt.subplots(1)
ax.imshow(Y[:,:,1],cmap='pink')
ax.plot(X[:n,0],X[:n,1],'r.',markersize=10,alpha=0.3)
ax.plot(X[n:,0],X[n:,1],'g.',markersize=10,alpha=0.3)
ax.set_aspect('equal')

#%% Now with mini batches
nl = [x.shape[1], 50,50,50,50, 2]

batch_size = 10

W = init_weights(nl)
fig, ax = plt.subplots(1)
n_iter = 21
L = np.zeros((n_iter))
i_rng = np.arange(0,n_iter) 
for i in range(0,n_iter):
    nb = Xc.shape[0]
    idx = np.random.permutation(nb)
    for j in range(0,nb,batch_size):
        Xb = Xc[idx[j:j+batch_size],:]
        Tb = T[idx[j:j+batch_size],:]
        W, l = backward(Xb,W,Tb,learning_rate=0.01,show_learning=True)
        L[i] += l
    if ( i%10 == 0):
        ax.cla()
        ax.plot(i_rng,L,'k')
        ax.set_title('Loss')
        plt.pause(0.001)
        plt.show()


y = forward(xc,W)[0]

Y = y.reshape((100,100,2))
fig,ax = plt.subplots(1)
ax.imshow(Y[:,:,1],cmap='pink')
ax.plot(X[:n,0],X[:n,1],'r.',markersize=10,alpha=0.3)
ax.plot(X[n:,0],X[n:,1],'g.',markersize=10,alpha=0.3)
ax.set_aspect('equal')







































# %%
