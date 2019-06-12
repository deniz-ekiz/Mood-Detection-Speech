import librosa
import os
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
import numpy as np
from tqdm import tqdm

max_len=11
path="/home/deniz/Desktop/RNN/Ses-Tanıma-deniz"
label="demo"


mfcc_vectors = []

wavfiles = [path + label + '/' + wavfile for wavfile in os.listdir(path + '/' + label)]
for wavfile in tqdm(wavfiles, "Saving vectors of label - '{}'".format(label)):
        wave, sr = librosa.load("/home/deniz/Desktop/RNN/Ses-Tanıma-deniz/demo/demo.wav", mono=True, sr=None)
        # wave = wave[::3]
        mfcc = librosa.feature.mfcc(wave, sr=16000, n_mfcc=20)

        if (max_len > mfcc.shape[1]):
            pad_width = max_len - mfcc.shape[1]
            mfcc = np.pad(mfcc, pad_width=((0, 0), (0, pad_width)), mode='constant')

        # Else cutoff the remaining parts
        else:
            mfcc = mfcc[:, :max_len]
        mfcc_vectors.append(mfcc)
np.save(label + '.npy', mfcc_vectors)



