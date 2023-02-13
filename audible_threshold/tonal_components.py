import numpy as np


def Dk_Sparse(K_max):
    D = np.zeros((K_max, K_max))
    D[2:281, 2] = 1
    D[281:569, 2:13] = 1
    D[569:1151, 2:27] = 1
    return D


def ST_init(c, D):
    
    ...
