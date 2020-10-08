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
        for index, name in enumerate(files):
            filename = os.path.join(root, name)
            if filename == './polyushka.wav':
                audio_all, sr = librosa.load(filename, sr = 16000)
                spectrogram_all = librosa.stft(audio_all, n_fft=2048, hop_length=256, center=False, win_length=2048) 
                N = abs(spectrogram_all)
                phase_all = N/(N + 2.2204e-16)

                middle = N[:, int(math.ceil(N.shape[1]/2))]
                middle[middle < (max(middle)/100)] = 0
                middle = middle/np.linalg.norm(middle)
                
                Matrix[index] = middle 
                Matrix_inv = np.linalg.pinv(Matrix)  # Matrix is N
                W = np.matmul(Matrix_inv.T,M)
    return Matrix, W


def create_E(M, Matrix, W):
    weights = []
    errors=np.zeros((1000,4))
    W = np.zeros(((Matrix.T).shape[1],M.shape[1]))
    learning_rates = [100,1000,10000,100000]
    D = len(M)
    T = len(M[0])
    DT = np.dot(D,T)
    Divide = 1/DT


    for j in range(len(learning_rates)):
        W = np.zeros(((Matrix.T).shape[1],M.shape[1]))
        for i in range(1000):
            Matrix_times_W_minus_M = (Matrix.T @ W)-M
            Matrix_times_Matrix_times_W_minus_M = Matrix @ Matrix_times_W_minus_M
            gradient_descent = Divide * 2 * Matrix_times_Matrix_times_W_minus_M
            W = W - float(learning_rates[j]) * gradient_descent
            W[W<0] = 0
            N_and_W = np.matmul(Matrix.T, W)

            # gradient_descent= 2 *(Matrix) @ (Matrix.T@W-M)/DT
            # W=W-float(learning_rates[j])*gradient_descent
            # W[W<0]=0
            # N_and_W=np.matmul(Matrix.T,W)

            errors[i,j] = np.linalg.norm(M - N_and_W)
            # print('errors:',errors)
        weights.append(W)
    return errors,weights


def plotting_error(errors,M,weights):
    D = len(M)
    T = len(M[0])
    DT = np.dot(D,T)
    Divide = 1/DT
    error=errors/DT
    # print(error)
    final_error=error[-1,:]
    print(final_error)
    index=np.argmin(final_error)
    final_W = weights[index]
    np.savetxt("problem2W.csv", final_W, delimiter=",")

    for i in range(4):
        learning_rates = [100,1000,10000,100000]
        plt.plot(np.arange(1000),error[:,i], label='Learning rate = {}'.format(learning_rates[i]))
        plt.ylabel('Error values')
        plt.xlabel('Iterations')
        plt.title("Error for each iteration number")
        plt.legend(loc='best')
    plt.show()  
    return final_W

        
if __name__ == "__main__":
    audio, sr = librosa.load('polyushka.wav', sr = 16000)
    spectrogram = librosa.stft(audio, n_fft=2048, hop_length=256, center=False, win_length=2048) 
    M = abs(spectrogram)
    phase = M/(M + 2.2204e-16)

    Matrix, W = create_spectograms()
    errors, weights = create_E(M, Matrix, W)
    final_W = plotting_error(errors,M,weights)

    M_hat = np.matmul(Matrix,final_W)
    signal_hat=librosa.istft(np.multiply(M_hat,phase),hop_length=256,center=False,win_length=2048)
    librosa.output.write_wav("resynthesized_nnproj.wav",signal_hat,sr=16000)
    

    



    