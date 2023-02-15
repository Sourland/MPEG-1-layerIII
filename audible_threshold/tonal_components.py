import numpy as np
from DCT.dct import DCT_power


def Dk_Sparse(K_max):
    D = np.zeros((K_max, K_max))
    D[2:281, 2] = 1
    D[281:569, 1:12] = 1
    D[569:1151, 1:26] = 1
    return D


def ST_init(c: np.ndarray, D):
    m, n = c.shape
    P = DCT_power(c)
    P = np.ndarray.flatten(P)
    tonal_components = []

    for k in range(1, m * n - 1):
        neighbors_right = k + np.where(D[k, :] == 1)[0]
        neighbors_left = k - np.where(D[k, :] == 1)[0]
        neighbors = np.array([neighbors_right, neighbors_left])

        if P[k] > P[k + 1] and P[k] > P[k - 1]:
            tonal_components.append(k)
        elif np.all(P[k] > (P[neighbors] + 7)):
            tonal_components.append(k)

    return tonal_components


def mask_power(c, ST):
    P = DCT_power(c)
    P_M = []
    for masker_idx in ST:
        exponential = 0
        for j in range(-1, 2):
            exponential += np.power(10, 0.1 * P(masker_idx + j))
        P_M.append(10*np.log10(exponential))

    return P_M


def ST_reduction(ST, c, Tq):
    P = DCT_power(c)
    for k in ST:
        if P[k] < Tq[k]:
            ST.delete(k)

    return ST, P[ST]


def spread_function(ST, PM, Kmax):
    ...