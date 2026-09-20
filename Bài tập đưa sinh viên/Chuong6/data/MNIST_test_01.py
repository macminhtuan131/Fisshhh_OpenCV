from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import time

print(tf.config.list_physical_devices())

(train_X, train_y), (test_X, test_y) = mnist.load_data()

print('X_train', train_X.shape)
print('y_train', train_y.shape)
print('X_test', test_X.shape)
print('y_test', test_y.shape)

model = Sequential()
model.add(Flatten(input_shape=(28, 28)))
model.add(Dense(units=2048, activation='sigmoid'))
model.add(Dense(units=4096, activation='sigmoid'))
model.add(Dense(10, activation='softmax'))
model.compile(optimizer='SGD', loss='sparse_categorical_crossentropy', metrics=['acc'])

model.summary()

start_time = time.time()
model_history = model.fit(train_X, train_y, 
                          validation_split=0.2,
                          epochs=10, batch_size=64)
stop_time = time.time()
execute_time = stop_time - start_time
print(f'Execute Time: {execute_time: .2f} seconds')

acc = model_history.history['acc']
val_acc = model_history.history['val_acc']
loss = model_history.history['loss']
val_loss = model_history.history['val_loss']

epochs = range(1, len(acc)+1)

plt.figure(figsize=(8, 8))
plt.subplot(2, 1, 1)
plt.plot(epochs, acc, 'b-o', label='Training acc')
plt.plot(epochs, val_acc, 'r--x', label='Validation acc')
plt.legend()
plt.grid()
plt.title('Traning & Validation accuracy')

plt.subplot(2, 1, 2)
plt.plot(epochs, loss, 'b-o', label='Training loss')
plt.plot(epochs, val_loss, 'r--x', label='Validation loss')
plt.legend()
plt.grid()
plt.title('Traning & Validation loss')
plt.show()

model.evaluate(x=test_X, y=test_y)