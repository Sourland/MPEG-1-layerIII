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
    bits_per_band_all_frames = []
    scale_factors_all_frames = []
    huff_total = []
    for i in range(MAX_LOOP//15):
        if i % 25 == 0:
            print(f"Encoding progress: " + "{:.2f}".format(i / (MAX_LOOP//10)) + "%")
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
        huff_total.append(huff_enc)

    return probabilities, huff_total, bits_per_band_all_frames, scale_factors_all_frames


def MP3_decoder(probabilites, frame_huff_sequence, scale_factors_all_frames, bits_per_band_all_frames):
    M, N = 32, 36
    Y_tot = np.array([])

    for idx, (prob, length, scale_factors, bits_per_band) in enumerate(
            zip(probabilites, frame_huff_sequence, scale_factors_all_frames,
                bits_per_band_all_frames)):

        encoded_sequence = frame_huff_sequence[idx]
        rle_symbols = huffman_decode(encoded_sequence, prob)
        symbols = run_length_decode(np.asarray(rle_symbols), len(rle_symbols))
        dct_coeffs = all_bands_dequantizer(symbols, bits_per_band, scale_factors)
        Yc = iframeDCT(dct_coeffs, N, M)
        if Y_tot.shape[0] == 0:
            Y_tot = Yc
        else:
            Y_tot = np.vstack((Y_tot, Yc))

    return decoder(Y_tot, h, M, N)


h = np.load("h.npy", allow_pickle=True).tolist()["h"]
M, N = 32, 36
L = 512
samplerate, wavin = wavfile.read("myfile.wav")
print(len(wavin))
prob, huff_bits, bits, scales = MP3_coder(wavin, h)
signal = MP3_decoder(prob, huff_bits, scales, bits)
print(len(signal))
wavfile.write("decoded_myfileMP3.wav", samplerate, signal.astype(np.int16))
