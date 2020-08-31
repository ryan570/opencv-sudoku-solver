from pathlib import Path

import numpy as np
import tensorflow as tf

path = Path(__file__).parent / 'new.model'
model = tf.keras.models.load_model(str(path.resolve()))

def predict(img):
    return np.argmax(model.predict(np.array([img])))
