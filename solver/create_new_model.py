from time import time
import os

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from keras.callbacks import TensorBoard
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers

img_width, img_height = 32, 32
num_train = 361
num_validation = 98
train_data_dir = 'data/train'
validation_data_dir = 'data/validation'
epochs = 10
batch_size = 32

model = tf.keras.models.Sequential()

model.add(layers.Conv2D(filters=6, kernel_size=(3, 3),
                        activation='relu', input_shape=(32, 32, 1)))
model.add(layers.AveragePooling2D())
model.add(layers.Conv2D(filters=16, kernel_size=(3, 3), activation='relu'))
model.add(layers.AveragePooling2D())
model.add(layers.Flatten())
model.add(layers.Dense(units=120, activation='relu'))
model.add(layers.Dense(units=84, activation='relu'))
model.add(layers.Dense(units=10, activation='softmax'))

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy']
              )

train_datagen = ImageDataGenerator(
    shear_range=0.2,
    zoom_range=[0.8, 1.2]
)

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    color_mode='grayscale',
    target_size=(img_width, img_height),
    batch_size=batch_size,
)

validation_generator = train_datagen.flow_from_directory(
    validation_data_dir,
    color_mode='grayscale',
    target_size=(img_width, img_height),
    batch_size=batch_size
)

steps_per_epoch = num_train // batch_size
validation_steps = num_validation // batch_size

tensorboard = TensorBoard(log_dir="logs/{}".format(time()))
model.fit(train_generator, steps_per_epoch=steps_per_epoch, epochs=epochs,
          validation_data=validation_generator, validation_steps=validation_steps, callbacks=[tensorboard])

# val_loss, val_acc = model.evaluate(train_generator)
# print("loss-> ", val_loss, "\nacc-> ", val_acc)

# path = os.path.dirname(os.path.dirname(__file__))
model.save('new2.model')
