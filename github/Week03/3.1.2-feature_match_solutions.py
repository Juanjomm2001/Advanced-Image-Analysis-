#%%
"""
Anders Bjorholm Dahl, abda@dtu.dk

Script to match images based on SIFT features.
"""



#%% Read two images to test the matching properties (default from documentation)
import numpy as np
import cv2
import matplotlib.pyplot as plt
import scipy.ndimage
import transform

path = './week3/week3/' # Replace with your own path

im1 = cv2.imread(path + 'CT_lab_high_res.png',cv2.IMREAD_GRAYSCALE) # queryImage
im2 = cv2.imread(path + 'CT_lab_low_res.png',cv2.IMREAD_GRAYSCALE) # trainImage

# Detecta puntos característicos en ambas imágenes

# Initiate SIFT detector
sift = cv2.SIFT_create()
kp1, des1 = sift.detectAndCompute(im1,None)
kp2, des2 = sift.detectAndCompute(im2,None)

# BFMatcher with default params
bf = cv2.BFMatcher()
# Para cada punto en imagen 1, encuentra los 2 puntos más similares en imagen 2
matches = bf.knnMatch(des1, des2, k=2)



# Aplica criterio de Lowe: si mejor_coincidencia/segunda_mejor < 0.6, es fiable

# Apply ratio test
good = []
for m,n in matches:
    # Apply the Lowe criterion best should be closer than second best
    if m.distance/(n.distance + 10e-10) < 0.6:
        good.append([m])

# cv.drawMatchesKnn expects list of lists as matches.
img3 = cv2.drawMatchesKnn(im1,kp1,im2,kp2,good,None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
plt.imshow(img3),plt.show()


#%% Match SIFT - function to extract the coordinates of matching points

# Parameters that can be extracted from keypoints
# keypoint.pt[0],
# keypoint.pt[1],
# keypoint.size,
# keypoint.angle,
# keypoint.response,
# keypoint.octave,
# keypoint.class_id,

def match_SIFT(im1, im2, thres = 0.6):
    
    # Initiate SIFT detector
    sift = cv2.SIFT_create()
    
    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(im1,None)
    kp2, des2 = sift.detectAndCompute(im2,None)
    
    # BFMatcher with default params
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1,des2,k=2)
    
    # Apply ratio test
    good_matches = []
    
    for m,n in matches:
        if m.distance/(n.distance+10e-10) < thres:
            good_matches.append([m])
    
    #Extrae coordenadas de puntos emparejados para calcular transformación


    pts_im1 = [kp1[m[0].queryIdx].pt for m in good_matches]
    pts_im1 = np.array(pts_im1, dtype=np.float32).T
    pts_im2 = [kp2[m[0].trainIdx].pt for m in good_matches]
    pts_im2 = np.array(pts_im2, dtype=np.float32).T
    return pts_im1, pts_im2






#%% Create a test image - Rotate, scale and crop image

ang = 67
sc = 0.6
imr = scipy.ndimage.rotate(scipy.ndimage.zoom(im1,sc),ang,reshape=False)[50:-50,50:-50]
plt.imshow(imr)

#%% Plot the keypoints with very low threshold (0.1) to see what is going on

pts_im1, pts_im2 = match_SIFT(im1, imr, 0.1)

def plot_matching_keypoints(im1, im2, pts_im1, pts_im2):
    r1,c1 = im1.shape
    r2,c2 = im2.shape
    n_row = np.maximum(r1, r2)
    n_col = c1 + c2
    im_comp = np.zeros((n_row,n_col))
    im_comp[:r1,:c1] = im1
    im_comp[:r2,c1:(c1+c2)] = im2
    
    fig,ax = plt.subplots(1)
    ax.imshow(im_comp, cmap='gray')
    ax.plot(pts_im1[0],pts_im1[1],'.r')
    ax.plot(pts_im2[0]+c1,pts_im2[1],'.b')
    ax.plot(np.c_[pts_im1[0],pts_im2[0]+c1].T,np.c_[pts_im1[1],pts_im2[1]].T,'w',linewidth = 0.5)

plot_matching_keypoints(im1,imr,pts_im1, pts_im2)

#%% Plot the keypoints between image 1 and image 2

pts_im1, pts_im2 = match_SIFT(im1, im2, 0.6)
plot_matching_keypoints(im1,im2,pts_im1, pts_im2)

#%% Compute the transformaions and plot them on top of each other

R,t,s = transform.get_transformation(pts_im2,pts_im1)

print(f'Transformations: Rotation:\n{R}\n\nTranslation:\n{t}\n\nScale: {s}\n\nAngle: {np.arccos(R[0,0])/np.pi*180}')
pts_im1_1 = s*R@pts_im2 + t

fig,ax = plt.subplots(1)
ax.imshow(im1)
ax.plot(pts_im1[0],pts_im1[1],'.r')
ax.plot(pts_im1_1[0],pts_im1_1[1],'.b')

#%% Robust transformation - kept points are shown in cyan and magenta

R,t,s,idx = transform.get_robust_transformation(pts_im2,pts_im1)

pts_im1_2 = s*R@pts_im2 + t

fig,ax = plt.subplots(1)
ax.imshow(im1, cmap = 'gray')
ax.plot(pts_im1[0],pts_im1[1],'.b')
ax.plot(pts_im1_2[0],pts_im1_2[1],'.r')
ax.plot(pts_im1[0,idx],pts_im1[1,idx],'.c')
ax.plot(pts_im1_2[0,idx],pts_im1_2[1,idx],'.m')





#%% Compute the blobs in the two images and match them (optional part)
# I have read in two images, that I will match using SIFT, compute 
# blobs based on the exercise week 2, and compare the matched set 
# of blobs.

import blob

#%% Compute the transformation of detected fibers in one image to the other

im1 = cv2.imread(path + 'CT_lab_high_res.png',cv2.IMREAD_GRAYSCALE) # queryImage
im2 = cv2.imread(path + 'CT_lab_med_res.png',cv2.IMREAD_GRAYSCALE) # trainImage

pts_im1, pts_im2 = match_SIFT(im1, im2, 0.6)
plot_matching_keypoints(im1,im2,pts_im1, pts_im2)


R,t,s,idx = transform.get_robust_transformation(pts_im2,pts_im1)

pts_im1_3 = s*R@pts_im2 + t

fig,ax = plt.subplots(1)
ax.imshow(im1, cmap = 'gray')
ax.plot(pts_im1[0],pts_im1[1],'.b')
ax.plot(pts_im1_3[0],pts_im1_3[1],'.r')
ax.plot(pts_im1[0,idx],pts_im1[1,idx],'.c')
ax.plot(pts_im1_3[0,idx],pts_im1_3[1,idx],'.m')

#%% Detect blobs in image 1

# Radius limit
diameterLimit = np.array([10,25])
stepSize = 0.3

# Parameter for Gaussian to detect center point
tCenter = 20

# Parameter for finding maxima over Laplacian in scale-space
thresMagnitude = 8

# Detect fibres
coord1, scale1 = blob.detectFibers(im1.astype(np.float32), diameterLimit, stepSize, tCenter, thresMagnitude)

# Plot detected fibres
fig, ax = plt.subplots(1,1,figsize=(10,10),sharex=True,sharey=True)
ax.imshow(im1, cmap='gray')
ax.plot(coord1[:,1], coord1[:,0], 'r.')
circ1_x, circ1_y = blob.getCircles(coord1, scale1)
plt.plot(circ1_x, circ1_y, 'r')
plt.show()
#%% Detect blobs in image 2

# Radius limit
diameterLimit = np.array([6,15])
stepSize = 0.15

# Parameter for Gaussian to detect center point
tCenter = 6

# Parameter for finding maxima over Laplacian in scale-space
thresMagnitude = 10

# Detect fibres
coord2, scale2 = blob.detectFibers(im2.astype(np.float32), diameterLimit, stepSize, tCenter, thresMagnitude)

# Plot detected fibres
fig, ax = plt.subplots(1,1,figsize=(10,10),sharex=True,sharey=True)
ax.imshow(im2, cmap='gray')
ax.plot(coord2[:,1], coord2[:,0], 'r.')
circ2_x, circ2_y = blob.getCircles(coord2, scale2)
plt.plot(circ2_x, circ2_y, 'r')
plt.show()

#%% Transform the coordinates

# I have the blob detection as (row,col), but SIFT is as (x,y), so I need to reverse t
coord2_to_1 = (s*R@coord2.T + t[[1,0]]).T 
scale2_to_1 = s*s*scale2.T
circ2_to_1_x, circ2_to_1_y = blob.getCircles(coord2_to_1, scale2_to_1)


# Plot detected fibres
fig, ax = plt.subplots(1,1,figsize=(10,10),sharex=True,sharey=True)
ax.imshow(im1, cmap='gray')
ax.plot(coord1[:,1], coord1[:,0], 'r.')
ax.plot(circ1_x, circ1_y, 'r')
ax.plot(coord2_to_1[:,1], coord2_to_1[:,0], 'g.')
ax.plot(circ2_to_1_x, circ2_to_1_y, 'b')
plt.show()

#%% Find match within n pixels

def find_nearest(p,q,dd=3):
    idx_pq = np.zeros((p.shape[0]), dtype=np.int_)
    d_pq = np.zeros((p.shape[0])) + 10e10
    idx_qp = np.zeros((q.shape[0]), dtype=np.int_)
    for i in range(p.shape[0]):
        d = np.sum((q-p[i,:])**2,axis=1)
        idx_pq[i] = np.argmin(d)
        d_pq[i] = d[idx_pq[i]]
    for i in range(q.shape[0]):
        d = np.sum((p-q[i,:])**2,axis=1)
        idx_qp[i] = np.argmin(d)
    
    p_range = np.arange(0,p.shape[0])
    match = idx_qp[idx_pq] == p_range
    idx_p = p_range[match*(d_pq < dd**2)]
    
    idx_q = idx_pq[match*(d_pq < dd**2)]
    return idx_p, idx_q

idx_1,idx_2_to_1 = find_nearest(coord1, coord2_to_1,5)

# Plot detected fibres
fig, ax = plt.subplots(1,1,figsize=(10,10),sharex=True,sharey=True)
ax.imshow(im1, cmap='gray')
ax.plot(coord1[idx_1,1], coord1[idx_1,0], 'r.')
ax.plot(circ1_x[:,idx_1], circ1_y[:,idx_1], 'r')
ax.plot(coord2_to_1[idx_2_to_1,1], coord2_to_1[idx_2_to_1,0], 'g.')
ax.plot(circ2_to_1_x[:,idx_2_to_1], circ2_to_1_y[:,idx_2_to_1], 'b')
plt.show()

#%% t-test for if the two distributions are different (they are)
import scipy.stats
print(scipy.stats.ttest_ind(scale1[idx_1], scale2_to_1[idx_2_to_1]))

min_val = np.min(np.r_[scale1[idx_1], scale2_to_1[idx_2_to_1]])
max_val = np.max(np.r_[scale1[idx_1], scale2_to_1[idx_2_to_1]])
h1 = np.histogram(scale1[idx_1], bins=40, range=(min_val,max_val))
h2 = np.histogram(scale2_to_1[idx_2_to_1], bins=40, range=(min_val,max_val))

# x = np.arange(min_val,max_val,step=(max_val-min_val)/50)
fig, ax = plt.subplots(1)
plt.title('Histograms of two distributions')
ax.bar((h1[1][1:]+h1[1][:-1])/2,h1[0], width = h1[1][1]-h1[1][0],color='r',alpha=0.3)
ax.plot((h1[1][1:]+h1[1][:-1])/2,h1[0],'r')
ax.bar((h2[1][1:]+h2[1][:-1])/2,h2[0], width = h2[1][1]-h2[1][0],color='b',alpha=0.3)
ax.plot((h2[1][1:]+h2[1][:-1])/2,h2[0],'b')
ax.legend(('CT large','CT medium to large'))


# %%
