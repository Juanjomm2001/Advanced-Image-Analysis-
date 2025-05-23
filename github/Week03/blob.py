#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Anders Bjorholm Dahl
abda@dtu.dk
2020
"""

import numpy as np
import skimage.feature
import cv2

# Gaussian derivatives

def getGaussDerivative(t):
    '''
    Computes kernels of Gaussian and its derivatives.
    Parameters
    ----------
    t : float
        Vairance - t.

    Returns
    -------
    g : numpy array
        Gaussian.
    dg : numpy array
        First order derivative of Gaussian.
    ddg : numpy array
        Second order derivative of Gaussian
    dddg : numpy array
        Third order derivative of Gaussian.

    '''

    kSize = 5
    s = np.sqrt(t)
    x = np.arange(int(-np.ceil(s*kSize)), int(np.ceil(s*kSize))+1)
    x = np.reshape(x,(-1,1))
    g = np.exp(-x**2/(2*t))
    g = g/np.sum(g)
    dg = -x/t*g
    ddg = -g/t - x/t*dg
    dddg = -2*dg/t - x/t*ddg
    return g, dg, ddg, dddg
    


# Show circles
def getCircles(coord, scale):
    '''
    Comptue circle coordinages

    Parameters
    ----------
    coord : numpy array
        2D array of coordinates.
    scale : numpy array
        scale of individual blob (t).

    Returns
    -------
    circ_x : numpy array
        x coordinates of circle. Each column is one circle.
    circ_y : numpy array
        y coordinates of circle. Each column is one circle.

    '''
    theta = np.arange(0, 2*np.pi, step=np.pi/100)
    theta = np.append(theta, 0)
    circ = np.array((np.cos(theta),np.sin(theta)))
    n = coord.shape[0]
    m = circ.shape[1]
    circ_y = np.sqrt(2*scale)*circ[[0],:].T*np.ones((1,n)) + np.ones((m,1))*coord[:,[0]].T
    circ_x = np.sqrt(2*scale)*circ[[1],:].T*np.ones((1,n)) + np.ones((m,1))*coord[:,[1]].T
    return circ_x, circ_y

# Function for detecting fibers
def detectFibers(im, diameterLimit, stepSize, tCenter, thresMagnitude):
    '''
    Detects fibers in images by finding maxima of Gaussian smoothed image

    Parameters
    ----------
    im : numpy array
        Image.
    diameterLimit : numpy array
        2 x 1 vector of limits of diameters of the fibers (in pixels).
    stepSize : float
        step size in pixels.
    tCenter : float
        Scale of the Gaussian for center detection.
    thresMagnitude : float
        Threshold on blob magnitude.

    Returns
    -------
    coord : TYPE
        DESCRIPTION.
    scale : TYPE
        DESCRIPTION.

    '''
    
    radiusLimit = diameterLimit/2
    radiusSteps = np.arange(radiusLimit[0], radiusLimit[1]+0.1, stepSize)
    tStep = radiusSteps**2/np.sqrt(2)
    
    r,c = im.shape
    n = tStep.shape[0]
    L_blob_vol = np.zeros((r,c,n))
    for i in range(0,n):
        g, dg, ddg, dddg = getGaussDerivative(tStep[i])
        L_blob_vol[:,:,i] = tStep[i]*(cv2.filter2D(cv2.filter2D(im,-1,g),-1,ddg.T) + 
                                      cv2.filter2D(cv2.filter2D(im,-1,ddg),-1,g.T))
    # Detect fibre centers
    g, dg, ddg, dddg = getGaussDerivative(tCenter)
    Lg = cv2.filter2D(cv2.filter2D(im, -1, g), -1, g.T)
    
    coord = skimage.feature.peak_local_max(Lg, threshold_abs = thresMagnitude)
    
    # Find coordinates and size (scale) of fibres
    magnitudeIm = np.min(L_blob_vol, axis = 2)
    scaleIm = np.argmin(L_blob_vol, axis = 2)
    scales = scaleIm[coord[:,0], coord[:,1]]
    magnitudes = -magnitudeIm[coord[:,0], coord[:,1]]
    idx = np.where(magnitudes > thresMagnitude)
    coord = coord[idx[0],:]
    scale = tStep[scales[idx[0]]]
    return coord, scale























