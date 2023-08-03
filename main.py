import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
import tensorflow as tf
from keras import models

# Load CIFAR-10 dataset
(training_images, training_labels), (testing_images, testing_labels) = tf.keras.datasets.cifar10.load_data()

# Normalize pixel values to range [0, 1]
training_images, testing_images = training_images / 255.0, testing_images / 255.0

class_names = ['Plane', 'Car', 'Bird', 'Cat', 'Deer', 'Dog', 'Frog', 'Horse', 'Ship', 'Truck']

# uncomment this to see the images in the data
# for i in range(16):
#     plt.subplot(4,4, i+1)
#     plt.xticks([])
#     plt.yticks([])
#     plt.imshow(training_images[i], cmap=plt.cm.binary)
#     plt.xlabel(class_names[training_labels[i][0]])

# plt.show()

# increse this value to get a more accurate result
training_images = training_images[:20000]
training_labels = training_labels[:20000]
testing_images = testing_images[:4000]
testing_labels = testing_labels[:4000]

# uncomment this to generate image classifier model
# model = models.Sequential()
# model.add(layers.Conv2D(32, (3,3), activation="relu", input_shape=(32,32,3)))
# model.add(layers.MaxPooling2D((2,2)))
# model.add(layers.Conv2D(64, (3,3), activation="relu"))
# model.add(layers.MaxPooling2D((2,2)))
# model.add(layers.Conv2D(64, (3,3), activation="relu"))
# model.add(layers.Flatten())
# model.add(layers.Dense(64, activation="relu"))
# model.add(layers.Dense(10, activation="softmax"))

# model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# model.fit(training_images, training_labels, epochs=10, validation_data=(testing_images, testing_labels))

# loss, accuracy = model.evaluate(testing_images, testing_labels)
# print(f"loss: {loss}")
# print(f"accuracy: {accuracy}")

# model.save('image_classifier.model')

model = models.load_model('image_classifier.model')

def preprocess_image(image):
    # Convert BGR to RGB
    img = cv.cvtColor(image, cv.COLOR_BGR2RGB)

    # Resize the image to 32x32
    img = cv.resize(img, (32, 32))

    # Normalize pixel values to range [0, 1]
    img = np.array([img]) / 255.0

    return img

def predict_image(image_path):
    # Read the image
    img = cv.imread(image_path)

    # Check if the image is already in the desired resolution (32x32)
    if img.shape[:2] != (32, 32):
        img = preprocess_image(img)

    # Make the prediction
    prediction = model.predict(img)
    index = np.argmax(prediction)
    return class_names[index]

# Replace 'IMAGE NAME' with the actual path of the image you want to predict
image_path = 'IMAGE NAME'
predicted_class = predict_image(image_path)
print(f'Prediction is {predicted_class}')

plt.show()