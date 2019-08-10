from keras.models import model_from_json
import numpy as np


giris=np.load("demo.npy")

EMOTIONS_LIST = ["alaycı","normal","sinirli"]

json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
loaded_model = model_from_json(loaded_model_json)

# load weights into the new model
loaded_model.load_weights("model.h5")
loaded_model._make_predict_function()

preds = loaded_model.predict(giris)
x=EMOTIONS_LIST[np.argmax(preds)]

print(x)
print(preds)