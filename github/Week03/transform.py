#esto es solo pa las funciones 





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

