"""
Anders Bjorholm Dahl, abda@dtu.dk

Script to develop function to estimate transformation parameters.
"""

import numpy as np

# Function that computes the parameters
def get_transformation(p, q):
    '''
    Compute the transformation parameters of the equation:
        q = s * R @ p + t

    Parameters
    ----------
    p : numpy array
        2 x n array of points.
    q : numpy array
        2 x n array of points. p and q corresponds.

    Returns
    -------
    R : numpy array
        2 x 2 rotation matrix.
    t : numpy array
        2 x 1 translation matrix.
    s : float
        scale parameter.

    '''
    m_p = np.mean(p,axis=1, keepdims=True)
    m_q = np.mean(q,axis=1, keepdims=True)
    s = np.linalg.norm(q - m_q, axis=0).sum() / np.linalg.norm(p - m_p, axis=0).sum()
    C = (q - m_q) @ (p - m_p).T
    U, S, V = np.linalg.svd(C)
    
    R_ = U @ V
    R = R_ @ np.array([[1, 0],[0, np.linalg.det(R_)]])
    
    t = m_q - s * R @ m_p
    return R, t, s

# Robust transformation

def get_robust_transformation(p, q, thres = 3):
    '''
    Compute the transformation parameters of the equation:
        q = s * R @ p + t

    Parameters
    ----------
    p : numpy array
        2 x n array of points.
    q : numpy array
        2 x n array of points. p and q corresponds.

    Returns
    -------
    R : numpy array
        2 x 2 rotation matrix.
    t : numpy array
        2 x 1 translation matrix.
    s : float
        scale parameter.
    idx : numpy array
        index of the points in p and q that are inliers in the robust match

    '''
    
    R,t,s = get_transformation(p, q)

    q_1 = s * R @ p + t
    d = np.linalg.norm(q - q_1, axis=0)
    idx = np.where(d < thres)[0]
    R,t,s = get_transformation(p[:,idx], q[:,idx])
    
    return R, t, s, idx

import numpy as np
import matplotlib.pyplot as plt




#% Generate and plot point sets
n = 20
p = np.random.randn(2,n)

angle = 76
theta = angle/180*np.pi
R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
t = np.array([[1], [-2]])
s = 0.6

q = s*R@p + t

fig,ax = plt.subplots(1)
ax.plot(p[0], p[1], 'r.')
ax.plot(q[0], q[1], 'b.')
ax.plot(np.c_[p[0], q[0]].T, np.c_[p[1], q[1]].T, 'g', linewidth=0.8)
ax.set_aspect('equal')

#% Compute parameters

m_p = np.mean(p,axis=1, keepdims=True)
m_q = np.mean(q,axis=1, keepdims=True)
s1 = np.linalg.norm(q - m_q, axis=0).sum() / np.linalg.norm(p - m_p, axis=0).sum()
print(s1)

C = (q - m_q) @ (p - m_p).T
U,S,V = np.linalg.svd(C)

R_ = U@V
R1 = R_@np.array([[1, 0], [0, np.linalg.det(R_)]])
print(R1)

t1 = m_q - s1 * R1 @ m_p
print(t1)

#%Function that computes the parameters

R2, t2, s2 = get_transformation(p, q)

print(f'Original: Rotation:\n{R}\n\nTranslation:\n{t}\n\nScale: {s}\n')
print(f'Computed: Rotation:\n{R2}\n\nTranslation:\n{t2}\n\nScale: {s2}')


#% Test the function wiht added noise to the q point set

sc = 0.1
qn = q + sc*np.random.randn(q.shape[0], q.shape[1])

R3,t3,s3 = get_transformation(p, qn)

print(f'Original: Rotation:\n{R}\n\nTranslation:\n{t}\n\nScale: {s}\n')
print(f'Computed: Rotation:\n{R3}\n\nTranslation:\n{t3}\n\nScale: {s3}')


#% Robust transformation

R4,t4,s4,idx = get_robust_transformation(p, qn, 0.1)
print(f'Original:    Rotation:\n{R}\n\nTranslation:\n{t}\n\nScale: {s}\n')
print(f'Transformed: Rotation:\n{R3}\n\nTranslation:\n{t3}\n\nScale: {s3}\n')
print(f'Robust:      Rotation:\n{R4}\n\nTranslation:\n{t4}\n\nScale: {s4}')
# Transformation gets sligthly better

fig,ax = plt.subplots(1)
ax.plot(p[0], p[1],'r.')
ax.plot(qn[0], qn[1],'b.')
ax.plot(p[0,idx], p[1,idx], 'm.', markersize=8)
ax.plot(qn[0,idx], qn[1,idx], 'c.', markersize=8)
ax.plot(np.c_[p[0], qn[0]].T, np.c_[p[1], qn[1]].T, 'g', linewidth=0.8)
ax.set_aspect('equal')



m = np.mean(p,axis=1, keepdims=True)

