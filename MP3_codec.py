from codec import coder, decoder
from dct import frameDCT, iframeDCT
from tonal_components import Dk_Sparse, psycho
from quantitizer import all_bands_quantizer, all_bands_dequantizer
from run_length_code import run_length_encode, run_length_decode
from huffman import huffman_encode, huffman_decode
import numpy as np
from scipy.io import wavfile


def MP3_codec(input, h):
    ...


def MP3_coder(input, h):
    M, N = 32, 36
    Y_tot = coder(input, h, M, N)
    MAX_LOOP = Y_tot.shape[0] // N
    probabilities = []
    frame_huff_length = []
    bits_per_band_all_frames = []
    scale_factors_all_frames = []
    f = open("encoded_steam.txt", "wb")
    for i in range(MAX_LOOP):
        if i % 10 == 0:
            print(i)
        frame = Y_tot[i * N:(i + 1) * N]
        dct_coefficients = frameDCT(frame)
        D = Dk_Sparse(M * N - 1)
        Tg = psycho(dct_coefficients, D)
        symbols, scale_factors, bits_per_band = all_bands_quantizer(dct_coefficients, Tg)
        bits_per_band_all_frames.append(bits_per_band)
        scale_factors_all_frames.append(scale_factors)
        rle = run_length_encode(symbols, len(symbols))
        rle_tuple = tuple([tuple(row) for row in rle])
        prob, huff_enc = huffman_encode(rle_tuple)
        probabilities.append(prob)
        frame_huff_length.append(huff_enc)
        f.write(bytes(huff_enc, 'utf-8'))

    f.close()
    return probabilities, frame_huff_length, bits_per_band_all_frames, scale_factors_all_frames


def MP3_decoder(probabilites, frame_huff_length, scale_factors_all_frames, bits_per_band_all_frames):
    M, N = 32, 36
    f = open("encoded_steam.txt", 'rb')
    sequence = f.read()
    reading_idx = 0
    Y_tot = zeros((N, M))
    for prob, length, scale_factors, bits_per_band in zip(probabilites, frame_huff_length, scale_factors_all_frames,
                                                          bits_per_band_all_frames):
        encoded_sequence = sequence[reading_idx:reading_idx + length]
        reading_idx += length
        rle_symbols = huffman_decode(np.asarray(encoded_sequence), prob)
        symbols = run_length_decode(rle_symbols, len(symbols))
        dct_coeffs = all_bands_dequantizer(symbols, bits_per_band, scale_factors)
        Yc = iframeDCT(dct_coeffs, N, M)
        Y_tot = np.vstack((decoded_signal, Yc))

    return decoder(Y_tot, h, M, N)


h = np.load("h.npy", allow_pickle=True).tolist()["h"]
M, N = 32, 36
L = 512
samplerate, wavin = wavfile.read("myfile.wav")
prob, huff_len, bits, scales = MP3_coder(wavin, h)
signal = MP3_decoder(prob, huff_len, bits, scales)
wavfile.write("decoded_myfile.wav", samplerate, decoded.astype(np.int16))
