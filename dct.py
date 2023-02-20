from scipy.fft import dct, idct
import numpy as np


def frameDCT(Y: np.ndarray):
    Y_dct = np.zeros(Y.shape)
    for i in range(Y.shape[0]):
        Y_dct[i] = dct(Y[i], norm="ortho")

    return Y_dct.flatten()


def iframeDCT(Y_dct, N, M):
    Y_dct = np.reshape(Y_dct, (N, M))
    Y = np.zeros(Y_dct.shape)
    for i in range(Y_dct.shape[0]):
        Y[i] = idct(Y_dct[i], norm="ortho")
    return Y


def DCT_power(dct_coefficients):
    norm = np.power(np.abs(dct_coefficients).astype('float64'), 2)
    return 10 * np.log10(norm)
