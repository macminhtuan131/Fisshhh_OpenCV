import tensorflow as tf
from tensorflow.keras.models import load_model
import cv2
import matplotlib.pyplot as plt
import numpy as np

model_filename = 'model_mnist_ver1.h5'
model = load_model(model_filename)

def ImageProcessing(img_filename):
    img = cv2.imread(img_name)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img1 = cv2.resize(img_gray, (28, 28), interpolation=cv2.INTER_LINEAR)
    img2 = 255 - img1
    return img2

img_name = './data_test/a2.png'
img_new = ImageProcessing(img_name)

plt.figure()
plt.subplot(1, 2, 1)
plt.imshow(cv2.cvtColor(cv2.imread(img_name), cv2.COLOR_BGR2RGB))
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(img_new, cmap='gray')
plt.axis('off')
plt.show()

data = np.expand_dims(img_new, axis=0)
result = model.predict(data)
predict = np.argmax(result)
print('Ket qua du doan anh:', predict)