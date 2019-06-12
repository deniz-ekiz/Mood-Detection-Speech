from keras.models import Sequential
from keras.layers import Dense
from keras.models import load_model
from keras.models import model_from_json
import numpy
import os
import librosa
import os
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
import numpy as np
from tqdm import tqdm
#from predicislem import *


giris=np.load("demo.npy")



json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("model.h5")
print("Loaded model from disk")

x=loaded_model.predict(giris)
y=np.argmax(loaded_model.predict(giris),axis=-1)

print(y)
print(loaded_model.predict(giris))
