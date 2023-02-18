import numpy as np


def Hz2Barks(f):
    return 13 * np.arctan(0.00076 * f) + 3.5 * np.arctan(np.power(f / 7500, 2))


def spread_function_value(Dz, k, P_M):
    value = 0

    if -3 <= Dz - 1:
        value = 17 * Dz - 0.4 * P_M[k] + 11
    elif -1 <= Dz < 0:
        value = Dz * (0.4 * P_M[K] + 6)
    elif 0 <= Dz < 1:
        value = -17 * Dz
    elif 1 <= Dz < 8:
        value = Dk * (0.15 * P_M[k] - 17) + 0.15 * P_M[k]

    return value


def dct_frequencies(coefficients, length):
    f_sampling = 44100
    return coefficients * f_sampling / (2 * length)
