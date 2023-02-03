from mp3 import make_mp3_analysisfb, make_mp3_synthesisfb
from frame import frame_sub_analysis, frame_sub_synthesis
from nothing import donothing, idonothing
from scipy.io import wavfile
import numpy as np


def codec(wavin, h, M, N, samplerate):
    L = 512
    Y_tot = coder(wavin, h, M, N)
    decoded_wavin = decoder(Y_tot, h, M, N)
    wavfile.write("../decoded_myfile.wav", samplerate, decoded_wavin)


def coder(wavin, h, M, N):
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
    H = make_mp3_analysisfb(h, M)

    padding_size = wavin.size % (M * N) + L
    padding = np.zeros(padding_size)
    sound_samples = np.concatenate((wavin, padding))

    Y_tot = np.empty((0, 1))

    for i in range(0, (sound_samples.size - L) // (M * N) - 1):
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

    decoded_wavin = np.empty((0, 1))

    for i in range(0, (Y_tot.shape[0] - L // M) // N - 1):
        Yc = Y_tot[i * N:(i + 1) * N + L // M]
        Yh = idonothing(Yc)
        Yh = frame_sub_synthesis(Yh, G)
        if Y_tot.shape[0] == 0:
            decoded_wavin = Yh
        else:
            decoded_wavin = np.append(decoded_wavin, Yh)

    return decoded_wavin


h = np.load("../h.npy", allow_pickle=True).tolist()["h"]
M, N = 32, 36
L = 512
samplerate, data = wavfile.read("../myfile.wav")

codec(data, h, M, N, samplerate)
