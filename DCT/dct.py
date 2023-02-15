from scipy.fft import dct, idct
import numpy as np


def frameDCT(Y: np.ndarray):
    Y_dct = np.zeros(Y.shape)
    for i in range(Y.shape[0]):
        Y_dct[i] = dct(Y[i])

    return Y_dct


def iframeDCT(Y_dct):
    Y = np.zeros(Y_dct.shape)
    for i in range(Y_dct.shape[0]):
        Y[i] = idct(Y_dct[i])
    return Y


def DCT_power(dct_coefficients):
    norm = np.power(np.abs(dct_coefficients), 2)
    return 10 * np.log10(norm)
