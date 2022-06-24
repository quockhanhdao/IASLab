import keras
import numpy as np
from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Flatten, Dense, Dropout
from keras.callbacks import ModelCheckpoint

# Load data (RGB Images: 32 x 32)
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
model.add(Flatten(input_shape = x_train.shape[1:]))
model.add(Dense(512, activation = 'relu'))
model.add(Dropout(0.2))
model.add(Dense(256, activation = 'relu'))
model.add(Dropout(0.2))
model.add(Dense(num_classes, activation = 'softmax'))

model.summary()

# Compile the model
model.compile(loss = 'categorical_crossentropy', 
              optimizer = 'rmsprop', metrics = ['accuracy'])

# Start training
hist = model.fit(x_train, y_train,
                batch_size = 32, epochs = 10,
                shuffle = True)

# Saving
model.save_weights('cifar10_mlp_model.hdf5')
print("Saving successfully")

# Loading
model.load_weights('cifar10_mlp_model.hdf5')
print("Loading successfully")

# Calculate accuracy
score = model.evaluate(x_test, y_test, verbose=0)
print('Test accuracy:', score[1])