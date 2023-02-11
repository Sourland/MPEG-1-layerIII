from scipy.fft import dct
import numpy as np


def frameDCT(Y):
    return dct(Y, norm='ortho')


def iframeDCT(Y):
    return idct(Y, norm='ortho')


def DCT_power(dct_coefficients):
    norm = np.power(np.abs(dct_coefficients), 2)
    return 10 * np.log10(norm)
