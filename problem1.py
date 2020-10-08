import librosa
import numpy as np 
import matplotlib.pyplot as plt
from librosa import display
import os
import math


def create_spectograms():
    dir_name = '.'
    Matrix = np.zeros((15,1025))
    for root, dirs, files in os.walk(dir_name, topdown=False):
        # print('Found directory: %s' % root)
        for index, name in enumerate(files):
            filename = os.path.join(root, name)
            if filename.endswith('.wav'):
                audio_all, sr = librosa.load(filename, sr = 16000)
                spectrogram_all = librosa.stft(audio_all, n_fft=2048, hop_length=256, center=False, win_length=2048) 
                N = abs(spectrogram_all)
                phase_all = N/(N + 2.2204e-16)

                middle = N[:, int(math.ceil(N.shape[1]/2))]
                middle[middle < (max(middle)/100)] = 0
                middle = middle/np.linalg.norm(middle)
                
                Matrix[index] = middle 
                Matrix_inv = np.linalg.pinv(Matrix)  # Matrix is N
                W = np.dot(Matrix_inv.T,M)
    return Matrix, W

def recompose_music(Matrix,W):
    W[W < 0] = 0
    M_new = np.dot(Matrix.T,W)
    error = np.linalg.norm(M - M_new)
    print(error)
    return M_new



if __name__ == "__main__":
    audio, sr = librosa.load('polyushka.wav', sr = 16000)
    spectrogram = librosa.stft(audio, n_fft=2048, hop_length=256, center=False, win_length=2048) 
    M = abs(spectrogram)
    phase = M/(M + 2.2204e-16)

    # print(M.shape)
    # Decibles = librosa.core.amplitude_to_db(M)
    # librosa.display.specshow(Decibles)
    # plt.figure()
    # plt.show()

    Matrix, W = create_spectograms()
    np.savetxt("problem1.csv", W, delimiter=",")
    M_new = recompose_music(Matrix, W)
    signal_new = librosa.istft(np.multiply(M_new, phase), hop_length=256, center=False, win_length=2048) 
    librosa.output.write_wav("resynthensized proj.wav", signal_new, sr=16000)


    




