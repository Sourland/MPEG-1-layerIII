from mp3 import make_mp3_analysisfb, make_mp3_synthesisfb
from frame import frame_sub_analysis, frame_sub_synthesis
from nothing import donothing, idonothing
from scipy.io import wavfile
from math import ceil
import numpy as np


def codec(wavin, h, M, N):
    Y_tot = coder(wavin, h, M, N)
    print(Y_tot.shape)
    return decoder(Y_tot, h, M, N)


def coder(wavin: np.ndarray, h: np.array, M: int, N: int) -> np.ndarray:
    """
    Reads M*N samples of waving, calculates each Y frame of Y_tot
    Args:
        wavin: The samples of the file to be encoded
        h: The impulse response of the standard lowpass MPEG filter
        M: Total number of subbands the signal will be seperated to
        N: Number of samples contained in a sample collection

    Returns:
        Y_tot: The input sample buffer
    """
    L = 512
    H = make_mp3_analysisfb(h, M)

    number_of_frames = ceil(wavin.size / (M*N))

    padding_size = wavin.size % (M * N) + L
    padding = np.zeros(padding_size)
    sound_samples = np.concatenate((wavin, padding))

    Y_tot = np.empty((0, 1))

    for i in range(0, number_of_frames):
        buffer = sound_samples[i * M * N:(i + 1) * M * N + L]
        Y = frame_sub_analysis(buffer, H, N)
        Yc = donothing(Y)
        if Y_tot.shape[0] == 0:
            Y_tot = Yc
        else:
            Y_tot = np.vstack((Y_tot, Yc))

    return Y_tot


def decoder(Y_tot, h, M, N):
    """

     Args:
         file:
         h:
         M:
         N:
         L:

     Returns:

     """
    L = 512
    G = make_mp3_synthesisfb(h, M)

    number_of_frames = Y_tot.shape[0] // N

    padding_size = wavin.size % N + L // M
    padding = np.zeros((padding_size, Y_tot.shape[1]))
    padded_Y_tot = np.concatenate((Y_tot, padding))

    decoded_wavin = np.empty((0, 1))

    for i in range(0, number_of_frames):
        Yc = padded_Y_tot[i * N:(i + 1) * N + L // M]
        Yh = idonothing(Yc)
        Yh = frame_sub_synthesis(Yh, G)
        if Y_tot.shape[0] == 0:
            decoded_wavin = Yh
        else:
            decoded_wavin = np.append(decoded_wavin, Yh)

    return decoded_wavin


# h = np.load("h.npy", allow_pickle=True).tolist()["h"]
# M, N = 32, 36
# L = 512
# samplerate, wavin = wavfile.read("myfile.wav")
#
# decoded = codec(wavin, h, M, N)
#
# wavfile.write("decoded_myfile.wav", samplerate, decoded.astype(np.int16))
#
# print(wavin.size)
# print(decoded.size)
#
# shifted_wavin = wavin[L-M:]
# shifted_decoded = decoded[:-L+M]
#
# signal = np.mean(shifted_wavin**2)
# noise = np.mean((shifted_wavin - shifted_decoded)**2)
# print(signal)
# print(noise)
# SNR = 10*np.log(signal/noise)
#
# print(f"SNR = {SNR}")