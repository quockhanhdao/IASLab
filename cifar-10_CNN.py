import keras
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.callbacks import ModelCheckpoint
from keras.datasets import cifar10

# Load data
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# Scale
scale = 255
x_train = x_train.astype('float32') / scale
x_test = x_test.astype('float32') / scale

# One-hot encode the labels
num_classes = 10
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

# Model
model = Sequential()
model.add(Conv2D(filters = 16, kernel_size = 2,
                padding = 'same', activation = 'relu', input_shape = x_train.shape[1:]))
model.add(MaxPooling2D(pool_size=2))
model.add(Conv2D(filters = 128, kernel_size = 2, 
                 padding='same', activation = 'relu'))
model.add(MaxPooling2D(pool_size = 2))
model.add(Conv2D(filters = 64, kernel_size = 2, 
                 padding = 'same', activation = 'relu'))
model.add(MaxPooling2D(pool_size = 2))
model.add(Dropout(0.3))
model.add(Flatten())
model.add(Dense(500, activation = 'relu'))
model.add(Dropout(0.4))
model.add(Dense(10, activation = 'softmax'))

model.summary()

# Compile the model
model.compile(loss = 'categorical_crossentropy',
             optimizer = 'rmsprop', metrics = ['accuracy'])

# Start training
hist = model.fit(x_train, y_train, 
                 batch_size = 32, epochs = 5, 
                 shuffle = True)

# Saving
model.save_weights('cifar10_cnn_model.hdf5')
print("Saving successfully")

# Loading
model.load_weights('cifar10_cnn_model.hdf5')
print("Loading successfully")

# Calculate accuracy
score = model.evaluate(x_test, y_test, verbose=0)
print('Test accuracy:', score[1])