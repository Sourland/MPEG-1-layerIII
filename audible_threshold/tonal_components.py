import numpy as np
from DCT.dct import DCT_power, frameDCT, iframeDCT
from scipy.io import wavfile
from subband_filtering.frame import frame_sub_analysis
from subband_filtering.mp3 import make_mp3_analysisfb


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

        neighbors = np.delete(neighbors, np.where(neighbors < 0))
        neighbors = np.delete(neighbors, np.where(neighbors >= m * n))

        if P[k] > P[k + 1] and P[k] > P[k - 1]:
            tonal_components.append(k)
        if neighbors.size == 0:
            continue

        elif np.all(P[k] > (P[neighbors] + 7)):
            tonal_components.append(k)

    return np.unique(tonal_components)


def mask_power(c, ST):
    P = DCT_power(c)
    P = np.ndarray.flatten(P)
    P_M = []
    for masker_idx in ST:
        exponential = 0
        for j in range(-1, 2):
            exponential += np.power(10, 0.1 * P[masker_idx + j])
        P_M.append(10 * np.log10(exponential))

    return P_M


def ST_reduction(ST, c, Tq):
    P = DCT_power(c)
    P = np.ndarray.flatten(P)
    for k in ST:
        if P[k] < Tq[k]:
            ST = np.delete(ST, np.where(ST == k))

    return ST, P[ST]


def spread_function(ST, PM, Kmax):
    ...


h = np.load("../h.npy", allow_pickle=True).tolist()["h"]
M, N = 32, 36
L = 512
samplerate, wavin = wavfile.read("../myfile.wav")
frame = wavin[0:M * N + L]
H = make_mp3_analysisfb(h, M)
Yc = frame_sub_analysis(frame, H, N)
dct = frameDCT(Yc)
Dk = Dk_Sparse(M * N - 1)
ST = ST_init(Yc, Dk)
PM = mask_power(Yc, ST)
Tq = np.load("../Tq.npy", allow_pickle=True).tolist()[0]
ST, PEEE = ST_reduction(ST, Yc, Tq)
i = 0
