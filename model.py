import os

import numpy as np
import tensorflow as tf

# path = os.path.dirname(os.path.dirname(__file__))
# model = tf.keras.models.load_model(os.path.join(path, 'LeNet5.model'))
model = tf.keras.models.load_model('new.model')

def predict(img):
    # prediction = model.predict(np.array([img]))
    # prediction = np.reshape(prediction, (9,))
    # # print(prediction)
    # count = 0
    # for val in prediction:
    #     if val > 0.1:
    #         count += 1
    # return count <= 1, np.argmax(model.predict(np.array([img]))) + 1
    return np.argmax(model.predict(np.array([img])))
    # return model.predict(np.array([img]))

