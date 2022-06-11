from keras.models import Sequential
from keras.layers import Dense
from keras.models import model_from_json
import numpy
import os


def load_model():
    # load json and create model
    json_file = open('notebooks\model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights("notebooks\model.h5")
    print("Loaded model from disk")
    loaded_model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
    return loaded_model
