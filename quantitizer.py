import numpy as np
from helpers import dct_frequencies
from scipy.io import wavfile
from codec import coder
from dct import frameDCT
from tonal_components import psycho, Dk_Sparse
import matplotlib.pyplot as plt


def critical_bands(K):
    """
    Calculates the critical bands for a given number of frequency bins.

    Args:
        K (int): Number of frequency bins.

    Returns:
        numpy.ndarray: An array of integers representing the critical band number for each frequency bin.
    """
    # Define the upper limits of the critical bands.
    band_limits = np.array([100, 200, 300, 400, 510, 630,
                            770, 920, 1080, 1270, 1480, 1720,
                            2000, 2320, 2700, 3150, 3700,
                            4400, 5300, 6400, 7700, 9500,
                            12000, 15500])

    # Initialize an array to hold the critical band index for each frequency bin.
    cb = np.zeros(K)

    # Calculate the frequencies for each DCT index
    frequencies = dct_frequencies(np.array([i for i in range(K)]), K)

    # Calculate the critical band number for each frequency.
    for k, f in enumerate(frequencies):
        array = np.append(band_limits, f)
        cb[k] = array.argsort().argsort()[-1] + 1

    return cb.astype(int)


def DCT_band_scale(c: np.ndarray):
    """
       Applies a scaling rule to the DCT coefficients of a signal based on critical bands of human hearing.

       Args:
           c (numpy.ndarray): A 1D NumPy array of DCT coefficients.

       Returns:
           Tuple[numpy.ndarray, numpy.ndarray]: A tuple containing two 1D NumPy arrays.
               The first array contains the scaled DCT coefficients, and the second array
               contains the scale factors used to scale each coefficient.
       """
    # Calculate the critical band number for each frequency bin.
    cb = critical_bands(c.shape[0])

    # Apply the scaling rule to the DCT coefficients.
    scaling_rule = np.power(np.abs(c), 3 / 4)

    # Initialize arrays to hold the scaled DCT coefficients and the scale factors.
    c_tilde = np.zeros(c.shape)
    scale_factors = np.zeros(c.shape)

    # Scale the DCT coefficients for each critical band.
    for i in range(np.max(cb)):
        dct_coeffs_band = np.where(cb - 1 == i)[0]
        max_scale_factor = np.max(scaling_rule[dct_coeffs_band])
        sign = np.sign(c[dct_coeffs_band])
        c_tilde[dct_coeffs_band] = np.multiply(sign, scaling_rule[dct_coeffs_band]) / max_scale_factor
        scale_factors[dct_coeffs_band] = max_scale_factor

    # Return the scaled DCT coefficients and the scale factors as a tuple.
    return c_tilde, scale_factors


def quantizer(x: np.ndarray, b):
    if np.any(np.abs(x) == 1) and b == 1:
        return 0

    # Calculate the width of each quantizer zone
    w_b = 1 / (2 ** b)
    zone_deciders = np.array([])

    # Calculate the quantizer zones
    for i in range(-2 ** (b - 1), 2 ** (b - 1) + 1):
        zone_deciders = np.append(zone_deciders, 2 * i * w_b)

    # Delete the 0 value element from the zones to make it dead-zone
    zone_deciders = np.delete(zone_deciders[1:-1], np.where(zone_deciders[1:-1] == 0)[0])
    positions = np.zeros(x.shape[0])
    # Append the values to the zone boundaries and calculate their position
    for i in range(x.shape[0]):
        array = np.append(zone_deciders, x[i])
        positions[i] = array.argsort().argsort()[-1]

    return positions.astype(int) - (2 ** b - 1)


def dequantizer(synb_index, b):
    synb_index = synb_index + (2 ** b - 1)
    w_b = 1 / (2 ** b)
    zone_deciders = np.array([])

    for i in range(-2 ** (b - 1), 2 ** (b - 1) + 1):
        zone_deciders = np.append(zone_deciders, 2 * i * w_b)

    zone_deciders = np.delete(zone_deciders, np.where(zone_deciders[1:-1] == 0)[0])

    if b == 1:
        return (sum(zone_deciders[1:-1]) / 2) * np.ones(np.array([synb_index]).shape)

    return (zone_deciders[synb_index] + zone_deciders[synb_index + 1]) / 2


def all_bands_quantizer(c: np.ndarray, Tg: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Applies a quantization algorithm to a vector of DCT coefficients to find the optimal amounds of bits required for
    quantization using a set of threshold values defined by the psychoacoustic model.

    Args:
        c (np.ndarray): A vector of DCT coefficients.
        Tg (np.ndarray): A vector of psychoacoustic threshold values for each band.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: A tuple containing three elements:
            1. An array of quantized symbols of all DCT coefficients
            2. An array of scale factors for each band, used for rescaling coefficients.
            3. An array containing the optimal number of bits to use to quantize each coefficient
    """

    cb = critical_bands(c.shape[0])

    # Scales the DCT coefficients to be in the range [-1, 1]
    c_tilde, scale_factors = DCT_band_scale(c)

    # Initializes the number of bits for quantization for each coefficient starting from 1
    bits_per_band = np.ones(np.unique(cb).shape).astype(int)

    # Initializes an array to store the quantized symbols for each coefficient
    symbols_c = np.zeros(c.shape)

    # Loops over each band in c_tilde
    for i in np.unique(cb - 1):
        # Take the indexes for every critical band
        band_indexes = np.where(cb == i + 1)[0]

        # Chooses components, thresholds and scaling factors of current band
        band_components = c_tilde[band_indexes]
        band_thresholds = Tg[band_indexes]
        c_band = c[band_indexes]
        sfs = scale_factors[band_indexes]

        # Deletes elements where threshold is not a number (nan)
        band_components_reduced = np.delete(band_components, np.isnan(band_thresholds))
        c_band = np.delete(c_band, np.isnan(band_thresholds))
        sfs = np.delete(sfs, np.isnan(band_thresholds))
        band_thresholds = np.delete(band_thresholds, np.isnan(band_thresholds))

        while 1:
            # Quantizes the coefficients in the band using the current number of bits
            c_symbols = quantizer(band_components_reduced, bits_per_band[i])

            # Dequantizes the coefficients using the same number of bits to get an approximation of the original
            # coefficients
            c_tilde_hat = dequantizer(c_symbols, bits_per_band[i])

            # Rescales the dequantized coefficients to their original values
            c_hat = np.sign(c_tilde_hat) * np.power(np.abs(c_tilde_hat) * sfs, 4 / 3)

            # Calculates the error between the original coefficients and the rescaled dequantized coefficients
            e = np.abs(c_hat - c_band)

            # Calculates the power of the error
            Pb = 10 * np.log10(e ** 2)

            # If the power of the error is greater than the psychoacoustic threshold, increment the number of bits
            if np.any(Pb > band_thresholds):
                bits_per_band[i] += 1
            else:
                symbols_c[band_indexes] = quantizer(band_components, bits_per_band[i])
                break
    # Returns the optimally quantized symbols of the DCT coefficients, the scale factors, and the number of bits per
    # coefficient
    return symbols_c.astype(int), scale_factors, bits_per_band


def all_bands_dequantizer(symb_index: np.ndarray, B: np.ndarray, SF: np.ndarray) -> np.ndarray:
    """
    Dequantizes symbols across all critical bands.

    Args:
        symb_index (np.ndarray): 1-D array of quantized symbols.
        B (np.ndarray): 1-D array of bits used to quantize every critical band
        SF (np.ndarray): 1-D array of scaling factors for each critical band.

    Returns:
        np.ndarray: 1-D array of dequantized symbols.
    """
    # Determine the critical bands for each symbol
    cb = critical_bands(symb_index.shape[0])
    # Initialize array for dequantized symbols
    dequantized_c = np.zeros(symb_index.shape)

    # Dequantize symbols in each critical band
    for i in np.unique(cb - 1):
        # Take the indexes for every critical band
        band_indexes = np.where(cb == i + 1)[0]

        # Select symbols and scaling factors for the current critical band
        band_symbols, sfs = symb_index[band_indexes], SF[band_indexes]

        # Dequantize symbols using the band corresponding number of bits
        c_tilde_hat = dequantizer(band_symbols, B[i])

        # Apply inverse power scaling to the dequantized symbols
        c_hat = np.sign(c_tilde_hat) * np.power(np.abs(c_tilde_hat) * sfs, 4 / 3)

        # Store dequantized symbols in the corresponding indices
        dequantized_c[band_indexes] = c_hat

    # Return the array of dequantized symbols
    return dequantized_c


# h = np.load("h.npy", allow_pickle=True).tolist()["h"]
# M, N = 32, 36
# L = 512
# samplerate, wavin = wavfile.read("myfile.wav")
# Y_tot = coder(wavin, h, M, N)
# all_pb = []
# all_tg = []
# for pepe in range(4):
#     Yc = Y_tot[pepe * N:(pepe + 1) * N, :]
#     Y_dct = frameDCT(Yc)
#
#     c_hat, scales = DCT_band_scale(Y_dct)
    # for b in range(1, 17):
    #     x = np.random.rand(5)
    #     print(f"initial: {x}, b = {b}")
    #     x_quant = quantizer(x, b)
    #     # print(f"Symbols = {x_quant} for b = {b}\n")
    #     x = dequantizer(x_quant, b)
    #     print(f"Qd for b = {b}: {x}\n")
    #
    # D = Dk_Sparse(M * N - 1)
    # Tg = psycho(Y_dct, D)
    # symbols, sf, bits = all_bands_quantizer(Y_dct, Tg)
    # Y_dct_hat = all_bands_dequantizer(symbols, bits, sf)
    # Pb = 10 * np.log10(np.abs(Y_dct_hat - Y_dct) ** 2)
    # haha = 0
    # all_tg.append(Tg)
    # all_pb.append(Pb)

# plt.plot(all_tg[0])
# plt.plot(all_pb[0])
# plt.grid()
# plt.legend(['Tg', 'Pb'])
# plt.show()

# plt.plot(all_tg[1])
# plt.plot(all_pb[1])
# plt.grid()
# plt.legend(['Tg', 'all_pb'])

# plt.show()
# plt.plot(all_tg[2])
# plt.plot(all_pb[2])
# plt.grid()
# plt.legend(['Tg', 'Pb'])

# plt.show()
# plt.plot(all_tg[4])
# plt.plot(all_pb[4])
# plt.grid()
# plt.legend(['Tg', 'Pb'])


# plt.show()
