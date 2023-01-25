import numpy as np


def make_mp3_analysisfb(h: np.ndarray, M: int) -> np.ndarray:
    """

    Args:
        h: Impulse response of the standard low-pass filter.
        M: Τhe number of zones to split into.

    Returns:
        A LxM matrix, where L is the length of the filter response. Every column is the h[i] filter response

    """

    H = np.zeros([len(h), M], dtype=np.float32)
    for i in range(1, M + 1):
        n = np.arange(h.shape[0], dtype=np.int64)[:,np.newaxis]
        frequency = (2 * i - 1) * np.pi / (2.0 * M)
        phase = -(2 * i - 1) * np.pi / 4.0
        tmp = np.cos(frequency * n + phase)
        x = np.multiply(h, tmp)
        H[:, i - 1, np.newaxis] = x
    return H


def make_mp3_synthesisfb(h: np.ndarray, M: int) -> np.ndarray:
    """

    Args:
        h:
        M:

    Returns:

    """

    H = make_mp3_analysisfb(h, M)
    L = len(h)
    G = np.flip(H, axis=0)
    return G

