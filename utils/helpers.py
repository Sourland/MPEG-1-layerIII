import numpy as np


def Hz2Barks(f):
    return 13 * np.arctan(76 * 1e-5) + 3.5 * arctan(np.power(f / 7500, 2))

