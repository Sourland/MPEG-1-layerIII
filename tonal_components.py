import numpy as np
from dct import DCT_power, frameDCT, iframeDCT
from scipy.io import wavfile
from helpers import Hz2Barks, spread_function_value, dct_frequencies
from codec import coder


def Dk_Sparse(K_max: np.ndarray) -> np.ndarray:
    """
    Function to create a sparse matrix D of size (K_max, K_max) containing the neighbor frequency for every potential masker

    Args:
        K_max (int): The maximum discrete frequencies per frame

    Returns:
        np.ndarray: a sparse matrix D representing the frequency neighbors
    """
    D = np.zeros((K_max, K_max))
    D[2:281, 1] = 1
    D[281:569, 1:12] = 1
    D[569:1151, 1:26] = 1
    return D


def ST_init(c: np.ndarray, D: np.ndarray) -> np.ndarray:
    """
    Extract tonal components from a signal by identifying peaks in the signal's DCT power.

    Args:
        c (np.ndarray): A 1D array containing the DCT coefficients of the signal.
        D (np.ndarray): A 2D array representing the frequency neighbours of the DCT coefficients of the signal.

    Returns:
        np.ndarray: A 1D array of indices of the selected tonal components of the signal.
    """
    # Compute power of DCT coefficients and flatten to 1D array
    P = DCT_power(c).flatten()

    tonal_components = []

    for k in range(1, c.shape[0] - 1):
        # Find indices of neighbors to the right and left of k
        neighbors = np.concatenate([k + np.where(D[k, :] == 1)[0],
                                    k - np.where(D[k, :] == 1)[0]])
        # Remove indices outside the range [0, c.shape[0])
        neighbors = neighbors[(neighbors >= 0) & (neighbors < c.shape[0])]

        # Check if k is a local maximum and larger than its neighbors by at least 7 dB
        if (P[k] > P[k + 1]) and (P[k] > P[k - 1]) and np.all(P[k] > P[neighbors] + 7):
            tonal_components.append(k)

    return np.unique(tonal_components)


def mask_power(c: np.ndarray, ST: np.ndarray) -> np.ndarray:
    """
    Computes the power in each masker, based on the DCT power coefficients and a given set of tonal components.

    Args:
        c (np.ndarray): 1D Array of DCT power coefficients.
        ST (np.ndarray): 1D array of indices of the selected tonal components of the signal.

    Returns:
        np.ndarray: Array of power values for each masker.
    """
    P = DCT_power(c)
    P_M = []
    for masker_idx in ST:
        masker_power = np.sum([np.power(10, 0.1 * P[masker_idx + j]) for j in range(-1, 2)])
        P_M.append(10 * np.log10(masker_power))

    return np.array(P_M)


def ST_reduction(ST: np.ndarray, c: np.ndarray, Tq: list) -> tuple[np.ndarray, np.ndarray]:
    """
    Performs spectral reduction on tonal components based on the masking thresholds.

    Args:
        ST (np.ndarray): A 1-D numpy array of tonal components.
        c (np.ndarray): A 2-D numpy array of DCT coefficients.
        Tq (list): A list of masking thresholds.

    Returns:
        A tuple (reduced_ST, mask_power_reduced_ST) containing:
        - reduced_ST (np.ndarray): A numpy array of the reduced tonal components.
        - mask_power_reduced_ST (np.ndarray): A numpy array of the corresponding mask power of the reduced tonal components.
    """
    # Calculate the mask power of the tonal components and the DCT coefficients
    P_M = mask_power(c, ST)

    # Select the tonal components whose mask power is greater than the absolute threshold
    Tq_ST = np.array(Tq)[ST]
    mask_power_greater_than_Tq = P_M >= Tq_ST
    ST = ST[mask_power_greater_than_Tq]
    P_M = P_M[mask_power_greater_than_Tq]

    # Select tonal components that are separated by at least 0.5 Bark
    signal_frequencies = dct_frequencies(ST, c.shape[0])
    signal_barks = Hz2Barks(signal_frequencies)
    diff_signal_barks = np.abs(np.diff(signal_barks))
    reduced_ST = []
    for idx, st in enumerate(ST[:-1]):
        if diff_signal_barks[idx] > 0.5:
            reduced_ST.append(st)
        elif P_M[idx] > P_M[idx + 1]:
            reduced_ST.append(ST[idx])
        else:
            reduced_ST.append(ST[idx + 1])

    reduced_ST = np.unique(np.array(reduced_ST))

    # Return the reduced tonal components and their corresponding mask power
    return reduced_ST.astype(int), P_M[np.in1d(ST, reduced_ST)]


def spread_function(ST: np.ndarray, PM: np.ndarray, Kmax: int) -> np.ndarray:
    """
    Computes spread function (SF) for each frequency band given the selected tonal components and mask power.

    Args:
        ST (np.ndarray): 1D array of indices of the selected tonal components of the signal.
        PM (np.ndarray): Mask power values for each tonal component.
        Kmax (int): The maximum discrete frequencies per frame

    Returns:
        np.ndarray: 2D array representing the spread function values between each tonal component and frequency band.
    """
    frequencies = dct_frequencies(np.array([i for i in range(Kmax + 1)]), Kmax + 1)

    SF_k = len(ST)

    SF = np.zeros((Kmax + 1, SF_k))

    for i in range(SF.shape[0]):
        for k in range(SF.shape[1]):
            Dz = Hz2Barks(frequencies[i]) - Hz2Barks(frequencies[k])
            SF[i, k] = spread_function_value(Dz, PM[k])
    return SF


def masking_thresholds(ST: np.ndarray, PM: np.ndarray, Kmax: int) -> np.ndarray:
    """
    Calculates the values of the audibility threshold for the set of discrete frequencies.

    Args:
        ST (np.ndarray): D array of indices of the selected tonal components of the signal.
        PM (np.ndarray): 1D array representing the power of the signal in the subbands.
        Kmax (int): The maximum discrete frequencies per frame

    Returns:
        T_m (np.ndarray): A 2D array of size (Kmax + 1, len(ST)) representing the calculated
        masking threshold values for the subbands.
    """
    SF = spread_function(ST, PM, Kmax)
    frequencies = dct_frequencies(np.array([i for i in range(Kmax + 1)]), Kmax + 1)

    SF_k = len(ST)
    T_m = np.zeros((Kmax + 1, SF_k))
    for i in range(SF.shape[0]):
        for k in range(SF.shape[1]):
            T_m[i, k] = PM[k] - 0.275 * Hz2Barks(frequencies[k]) + SF[i, k] - 6.025

    return T_m


def global_masking_thresholds(T_i: np.ndarray, T_q: np.ndarray) -> np.ndarray:
    """
    Computes the global masking threshold for each subband based on the he values of the audibility threshold for
    the set of discrete frequencies of a frame and the audibility threshold of silence

    Args:
        T_i (np.ndarray): An array of shape (Kmax+1, L), where Kmax is the maximum index of the subbands
                          and L is the number of critical bands. T_i represents the individual masking
                          threshold for each subband and critical band.
        T_q (np.ndarray): An array of shape (Kmax+1,), representing the absolute threshold in quiet for
                          each subband.

    Returns:
        np.ndarray: An vector of length Kmax+1, representing the global masking threshold for each subband.
    """
    T_g = np.zeros(len(T_q))
    for i in range(len(T_g)):
        arg = 10 ** (0.1 * T_q[i]) + np.sum(10 ** (0.1 * T_i[i, :]))
        T_g[i] = 10 * np.log10(arg)

    return T_g


def psycho(dct_coefficients, subband_neighbors):
    """
    Applies the psychoacoustic model to determine the global masking threshold for audio data.

    Parameters:
        dct_coefficients (ndarray): A 1D numpy array containing the DCT coefficients of audio data.
        subband_neighbors (ndarray): A 2D numpy array containing the neighboring frequencies for each subband.

    Returns:
        ndarray: A 1D numpy array containing the global masking threshold for each subband.
    """
    # Calculate the tonal components
    ST = ST_init(dct_coefficients, subband_neighbors)

    # Load the tonal masking threshold values
    Tq = np.load("Tq.npy", allow_pickle=True).tolist()[0]

    # Reduce the spectral envelope and determine the point spread function
    ST, PST = ST_reduction(ST, Y_dct, Tq)

    # Calculate the threshold in quiet and the maskers
    t_i = masking_thresholds(ST, PST, M * N - 1)

    # Calculate the global masking threshold
    return global_masking_thresholds(t_i, Tq)


h = np.load("h.npy", allow_pickle=True).tolist()["h"]
M, N = 32, 36
L = 512
samplerate, wavin = wavfile.read("myfile.wav")

Y_tot = coder(wavin, h, M, N)
Yc = Y_tot[4 * N:5 * N, :]
Y_dct = frameDCT(Yc)
Yy = iframeDCT(Y_dct, N, M)
Dk = Dk_Sparse(M * N - 1)

tg = psycho(Y_dct, Dk)
haha = 0
