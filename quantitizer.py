import numpy as np
from helpers import dct_frequencies
from scipy.io import wavfile
from codec import coder
from dct import frameDCT
from tonal_components import psyco, Dk_Sparse


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
                            770, 920, 1080, 1480, 1720,
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
    scaling_rule = np.power(np.abs(c), 0.75)

    # Initialize arrays to hold the scaled DCT coefficients and the scale factors.
    c_tilde = np.zeros(c.shape)
    scale_factors = np.zeros(c.shape)

    # Scale the DCT coefficients for each critical band.
    for i in range(np.max(cb)):
        dct_coeffs_band = np.where(cb == i + 1)[0]
        max_scale_factor = np.max(scaling_rule[dct_coeffs_band])
        sign = np.sign(c[dct_coeffs_band])
        c_tilde[dct_coeffs_band] = np.multiply(sign, scaling_rule[dct_coeffs_band]) / max_scale_factor
        scale_factors[dct_coeffs_band] = max_scale_factor

    # Return the scaled DCT coefficients and the scale factors as a tuple.
    return c_tilde, scale_factors


def quantizer(x: np.ndarray, b):
    if np.abs(x) == 1 and b == 1:
        return 0

    w_b = 1 / (2 ** b - 1)
    zone_deciders = np.array([])

    for i in range(- (2 ** b - 1), 2 ** b, 2):
        zone_deciders = np.append(zone_deciders, i * w_b)

    zone_deciders = zone_deciders[1:-1]

    array = np.append(zone_deciders, x)
    position = array.argsort().argsort()[-1]
    quantized_x = position - b

    return quantized_x


def dequantizer(synb_index, b):
    w_b = 1 / (2 ** b - 1)
    zone_deciders = np.array([])
    for i in range(- (2 ** b - 1), 2 ** b, 2):
        zone_deciders = np.append(zone_deciders, i * w_b)

    if b > 1:
        return (zone_deciders[synb_index + b + 1] + zone_deciders[synb_index + b]) / 2
    else:
        return sum(zone_deciders) / 2


def all_bands_quantizer(c: np.ndarray, Tg: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
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
    # Scales the DCT coefficients to be in the range [-1, 1]
    c_tilde, scale_factors = DCT_band_scale(c)

    # Initializes the number of bits for quantization for each coefficient starting from 1
    bits_per_coeff = np.ones(c.shape).astype(int)

    # Initializes an array to store the quantized symbols for each coefficient
    symbols_c = np.zeros(c.shape)

    # Loops over each band in c_tilde
    for i in range(c_tilde.shape[0]):
        # Continuously quantizes the coefficients in the band until the power of the error does not exceed the psychoacoustic threshold
        while 1:
            # Quantizes the coefficients in the band using the current number of bits
            c_symbol = quantizer(c_tilde[i], bits_per_coeff[i])

            # Dequantizes the coefficients using the same number of bits to get an approximation of the original coefficients
            c_tilde_hat = dequantizer(c_symbol, bits_per_coeff[i])

            # Rescales the dequantized coefficients to their original values
            c_hat = np.sign(c_tilde_hat) * c_tilde_hat * np.power(scale_factors[i], 4 / 3)

            # Calculates the error between the original coefficients and the rescaled dequantized coefficients
            e = np.abs(c_hat - c[i])

            # Calculates the power of the error
            Pb = 10 * np.log10(e ** 2)

            # If the power of the error is greater than the psychoacoustic threshold, increment the number of bits
            if Pb > Tg[i]:
                bits_per_coeff[i] += 1
            else:
            # Quantize using the optimal amount of bits and save
                symbols_c[i] = quantizer(c_tilde[i], bits_per_coeff[i])
                break

    # Returns the optimally quantized symbols of the DCT coefficients, the scale factors, and the number of bits per coefficient
    return symbols_c, scale_factors, bits_per_coeff


def all_bands_dequantizer(symb_index: np.ndarray, B: np.ndarray, SF: np.ndarray) -> np.ndarray:
    c_dequantized = np.zeros(symb_index.shape)

    for i in range(symb_index.shape[0]):
        c_dequantized[i] = dequantizer(symb_index[i], B[i])

    return np.sign(c_dequantized) * c_dequantized * np.power(SF, 4 / 3)



h = np.load("h.npy", allow_pickle=True).tolist()["h"]
M, N = 32, 36
L = 512
samplerate, wavin = wavfile.read("myfile.wav")

Y_tot = coder(wavin, h, M, N)
Yc = Y_tot[:N, :]
Y_dct = frameDCT(Yc)

c_hat, scales = DCT_band_scale(Y_dct)
# x_quant = quantizer(c_hat, 2)
# x = dequantizer(x_quant, 3)
D = Dk_Sparse(M * N - 1)
Tg = psyco(Y_dct, D)
symbols_c, scale_factors, bits_per_coeff = all_bands_quantizer(Y_dct, Tg)
Yc_hat = all_bands_dequantizer(symbols_c, bits_per_coeff, scale_factors)
omgomg = 0
