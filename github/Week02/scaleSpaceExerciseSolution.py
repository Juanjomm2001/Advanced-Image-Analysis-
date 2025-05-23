#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Anders Bjorholm Dahl
abda@dtu.dk
"""




#%%  2.1.1 "Computing Gaussian and its second order derivative
# FILTROS GAUSSIANOS 

import numpy as np
import matplotlib.pyplot as plt
import skimage.io
import skimage.feature
import cv2
data_path = './week2/' # replace with your own data path

im = skimage.io.imread(data_path + 'test_blob_uniform.png').astype(np.float32)

fig, ax = plt.subplots(1,1,figsize=(10,10),sharex=True,sharey=True)
ax.imshow(im,cmap='gray')

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

    kSize = 5  # determina el tamaño del kernel como ±5σ
    s = np.sqrt(t)
    x = np.arange(int(-np.ceil(s*kSize)), int(np.ceil(s*kSize))+1)
    x = np.reshape(x,(-1,1))
    g = np.exp(-x**2/(2*t))
    g = g/np.sum(g)   #es el kernel gaussiano (filtro de suavizado)
    dg = -x/t*g  # es la primera derivada del gaussiano y viceversa
    ddg = -g/t - x/t*dg
    dddg = -2*dg/t - x/t*ddg
    return g, dg, ddg, dddg
    
g, dg, ddg, dddg = getGaussDerivative(3)
fig, ax = plt.subplots(1, 1, figsize=(10, 8), sharex=True, sharey=True)

# Agregar cada curva con un color distinto y etiqueta
ax.plot(g, 'b-', linewidth=3, label='Gaussiana')
ax.plot(dg, 'r-', linewidth=2, label='Primera derivada')
ax.plot(ddg, 'g-', linewidth=2, label='Segunda derivada')
ax.plot(dddg, 'm-', linewidth=2, label='Tercera derivada')

# Agregar leyenda, título y etiquetas de ejes
ax.legend(fontsize=12)
ax.set_title('Gaussiana y sus derivadas (t=3)', fontsize=14)
ax.set_xlabel('Posición', fontsize=12)
ax.set_ylabel('Valor', fontsize=12)
ax.grid(True, linestyle='--', alpha=0.7)

# Añadir una línea horizontal en y=0 para referencia
ax.axhline(y=0, color='k', linestyle='-', alpha=0.3)

plt.tight_layout()
plt.show()




#%% Convolve an image

t = 325
g, dg, ddg, dddg = getGaussDerivative(t)

#mete un filtro como para tal pero para rebajar el ruido
Lg = cv2.filter2D(cv2.filter2D(im, -1, g), -1, g.T)
fig, ax = plt.subplots(1,1,figsize=(10,10),sharex=True,sharey=True)
ax.imshow(Lg,cmap='gray')


#%% 2.1.2 Detecting blobs on one scale
# Usa float o np.float32 en lugar de np.float
im = skimage.io.imread(data_path + 'test_blob_uniform.png').astype(float)
# Alternativa: im = skimage.io.imread(data_path + 'test_blob_uniform.png').astype(np.float32)

Lxx = cv2.filter2D(cv2.filter2D(im, -1, g), -1, ddg.T)
Lyy = cv2.filter2D(cv2.filter2D(im, -1, ddg), -1, g.T)

L_blob = t*(Lxx + Lyy)

# how blob response
fig, ax = plt.subplots(1,1,figsize=(10,10),sharex=True,sharey=True)
pos = ax.imshow(L_blob, cmap='gray')
fig.colorbar(pos)



#%% Find regional maximum in Laplacian

# VALe esto busca max y min extremos respecto a sus vecinos respecto al umbralmagnitud:
magnitudeThres = 50

coord_pos = skimage.feature.peak_local_max(L_blob, threshold_abs=magnitudeThres) #max para los
coord_neg = skimage.feature.peak_local_max(-L_blob, threshold_abs=magnitudeThres) #min 
coord = np.r_[coord_pos, coord_neg]

# Show circles conforme a los maximos y min
theta = np.arange(0, 2*np.pi, step=np.pi/100)
theta = np.append(theta, 0)
circ = np.array((np.cos(theta),np.sin(theta)))
n = coord.shape[0]
m = circ.shape[1]

fig, ax = plt.subplots(1,1,figsize=(10,10),sharex=True,sharey=True)
ax.imshow(im, cmap='gray')
plt.plot(coord[:,1], coord[:,0], '.r')
circ_y = np.sqrt(2*t)*np.reshape(circ[0,:],(1,-1)).T*np.ones((1,n)) + np.ones((m,1))*np.reshape(coord[:,0],(-1,1)).T
circ_x = np.sqrt(2*t)*np.reshape(circ[1,:],(1,-1)).T*np.ones((1,n)) + np.ones((m,1))*np.reshape(coord[:,1],(-1,1)).T
plt.plot(circ_x, circ_y, 'r')

"""El parámetro t controla el tamaño de los blobs que queremos detectar. 
Si ajustamos t correctamente, los círculos dibujados coincidirán exactamente con los bordes de los objetos circulares en la imagen.
"""






#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#   2.1.3 Detecting blobs on multiple scales

# En escalas diferntes no siempr la misma T

im = skimage.io.imread(data_path + 'test_blob_varying.png').astype(np.float64)

t = 15 #por empezar con uno
g, dg, ddg, dddg = getGaussDerivative(t)

r,c = im.shape
n = 100
L_blob_vol = np.zeros((r,c,n))
tStep = np.zeros(n)

Lg = im
for i in range(0,n):
    tStep[i] = t*i
    L_blob_vol[:,:,i] = t*i*(cv2.filter2D(cv2.filter2D(Lg, -1, g), -1, ddg.T) + 
        cv2.filter2D(cv2.filter2D(Lg, -1, ddg), -1, g.T))
    Lg = cv2.filter2D(cv2.filter2D(Lg, -1, g), -1, g.T)

thres = 40.0
coord_pos = skimage.feature.peak_local_max(L_blob_vol, threshold_abs = thres)
coord_neg = skimage.feature.peak_local_max(-L_blob_vol, threshold_abs = thres)
coord = np.r_[coord_pos,coord_neg]

# Show circles
theta = np.arange(0, 2*np.pi, step=np.pi/100)
theta = np.append(theta, 0)
circ = np.array((np.cos(theta),np.sin(theta)))
n = coord.shape[0]
m = circ.shape[1]

fig, ax = plt.subplots(1,1,figsize=(10,10),sharex=True,sharey=True)
ax.imshow(im, cmap='gray')
plt.plot(coord[:,1], coord[:,0], '.r')
scale = tStep[coord[:,2]]          # Obtiene el valor t para cada blob
circ_y = np.sqrt(2*scale)*np.reshape(circ[0,:],(1,-1)).T*np.ones((1,n)) + np.ones((m,1))*np.reshape(coord[:,0],(-1,1)).T   # Dibuja círculos con radio proporcional a escala
circ_x = np.sqrt(2*scale)*np.reshape(circ[1,:],(1,-1)).T*np.ones((1,n)) + np.ones((m,1))*np.reshape(coord[:,1],(-1,1)).T
plt.plot(circ_x, circ_y, 'r')









#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# 2.1.4 Detecting blobs in real data (scale space)

# rango de diametros paa uqe solo detecte entre eso
d = np.arange(10, 24.5, step = 0.4)
tStep = np.sqrt(0.5)*((d/2)**2) # convert to scale t

# read image and take out a small part
im = skimage.io.imread(data_path + 'SEM.png').astype(np.float64)
im = im[200:500,200:500]

fig, ax = plt.subplots(1,1,figsize=(10,10),sharex=True,sharey=True)
ax.imshow(im, cmap='gray')

#% Compute scale space

r,c = im.shape
n = d.shape[0]
L_blob_vol = np.zeros((r,c,n))

for i in range(0,n):
    g, dg, ddg, dddg = getGaussDerivative(tStep[i])
    L_blob_vol[:,:,i] = tStep[i]*(cv2.filter2D(cv2.filter2D(im,-1,g),-1,ddg.T) + 
                                  cv2.filter2D(cv2.filter2D(im,-1,ddg),-1,g.T))

#% Find maxima in scale space
#busca los min con vlaor mayor a 30
thres = 30
coord = skimage.feature.peak_local_max(-L_blob_vol, threshold_abs = thres)

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

scale = tStep[coord[:,2]]
circ_x, circ_y = getCircles(coord[:,0:2], scale)
fig, ax = plt.subplots(1,1,figsize=(10,10),sharex=True,sharey=True)
ax.imshow(im, cmap='gray')
plt.plot(coord[:,1], coord[:,0], '.r')
plt.plot(circ_x, circ_y, 'r')


"""
RESULTADO NO MUY BUENO CON LAPLACIAN SCALE"""





#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# 2.1.5 Localize blobs - Example high resolution lab X-ray CT - find the coordinates 
# using Gaussian smoothing and use the scale space to find the scale 

# El código anterior tenía dificultades para detectar todas las fibras con 
# un solo método. Esta nueva aproximación combina:

# Detección de centros de fibras usando suavizado gaussiano
# Determinación del tamaño usando escala del Laplaciano

im = skimage.io.imread(data_path + 'CT_lab_high_res.png').astype(np.float64)/255

fig, ax = plt.subplots(1,1,figsize=(10,10),sharex=True,sharey=True)
ax.imshow(im, cmap='gray')

#% Set parameters
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
    coord : numpy array
        n x 2 array of coordinates with row and column coordinates in each column.
    scale : numpy array
        n x 1 array of scales t (variance of the Gaussian).

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
    Lg = cv2.filter2D(cv2.filter2D(im, -1, g), -1, g.T)  #Detecta los centros de fibras con un método diferente:
    
    coord = skimage.feature.peak_local_max(Lg, threshold_abs = thresMagnitude)
    

    #Para cada centro detectado, encuentra la escala que da la respuesta más fuerte
    #Esto determina el diámetro preciso de cada fibra
    
    # Find coordinates and size (scale) of fibres
    magnitudeIm = np.min(L_blob_vol, axis = 2)
    scaleIm = np.argmin(L_blob_vol, axis = 2)
    scales = scaleIm[coord[:,0], coord[:,1]]

    # Elimina detecciones débiles que podrían ser ruido
    magnitudes = -magnitudeIm[coord[:,0], coord[:,1]]
    idx = np.where(magnitudes > thresMagnitude)
    coord = coord[idx[0],:]
    scale = tStep[scales[idx[0]]]
    return coord, scale


#% Set parameters

# Radius limit
diameterLimit = np.array([10,25])
stepSize = 0.3

# Parameter for Gaussian to detect center point
tCenter = 20

# Parameter for finding maxima over Laplacian in scale-space
thresMagnitude = 8

# Detect fibres
coord, scale = detectFibers(im, diameterLimit, stepSize, tCenter, thresMagnitude)

# Plot detected fibres
fig, ax = plt.subplots(1,1,figsize=(10,10),sharex=True,sharey=True)
ax.imshow(im, cmap='gray')
ax.plot(coord[:,1], coord[:,0], 'r.')
circ_x, circ_y = getCircles(coord, scale)
plt.plot(circ_x, circ_y, 'r')


"""
# Diferencia Principal Entre los Dos Métodos

La diferencia clave entre ambos métodos está en **cómo detectan los centros de las fibras**:

## Método 2.1.4 (Anterior)
- Usa **directamente el Laplaciano** para detectar tanto la posición como el tamaño
- Busca mínimos en `-L_blob_vol` (el negativo del Laplaciano)
- Un solo paso: detecta posición y tamaño simultáneamente
- Problema: puede perder algunas fibras si no producen una respuesta fuerte en el Laplaciano

## Método 2.1.5 (Nuevo)
- **Separa la detección en dos pasos**:
  1. **Primero**: Detecta centros usando **solo suavizado gaussiano** (`Lg = cv2.filter2D(cv2.filter2D(im, -1, g), -1, g.T)`)
  2. **Después**: Determina el tamaño usando el Laplaciano en esas posiciones
- Ventaja: El suavizado gaussiano produce picos más claros en los centros de las fibras, haciéndolos más fáciles de detectar

Es como usar diferentes herramientas para diferentes tareas:
- El filtro gaussiano es mejor para encontrar "dónde están" las fibras
- El Laplaciano es mejor para medir "cuán grandes son" las fibras

Este enfoque híbrido produce resultados más robustos, especialmente en imágenes reales donde las fibras pueden tener contraste variable o superponerse parcialmente."""
























