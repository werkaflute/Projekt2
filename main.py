import random
import PIL
from PIL import Image
import numpy as np
import os

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers.experimental.preprocessing import Rescaling
from matplotlib import pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.neural_network import MLPClassifier

def augment_data(dataset):
    generated_dataset = []
    for i in range(len(dataset)):
        gen = datagen.flow(np.stack(np.array(dataset)[i:i + 1, 0]), np.stack(np.array(dataset)[i:i + 1, 1]), batch_size=1)
        for j in range(6):
            generated_image = gen.next()[0]
            generated_dataset.append(np.array([generated_image, dataset[i][1]]))
    return generated_dataset

def create_datagen_basic():
    shift = 0.05

    datagen = ImageDataGenerator(
        featurewise_center=False,
        featurewise_std_normalization=False,
        rotation_range=4,
        width_shift_range=shift,
        height_shift_range=shift,
        horizontal_flip=True,
        zca_whitening=False,
        fill_mode='reflect'
    )

    return datagen

def standardize(image):
    size = (256, 256)
    ret = image.resize(size)
    return ret


def normalize(dataset):
    dataset_np = np.array(dataset)
    images = np.array(dataset_np[:, 0])
    images = images / 255
    normalized_dataset = []
    for i in range(len(dataset)):
        normalized_dataset.append([images[i], dataset[i][1]])
    return normalized_dataset


def split_data(dataset):
    train_set = [dataset[i] for i in range(len(dataset)) if i % 10 != 0 and i % 10 != 1]
    val_set = dataset[0::10]
    test_set = dataset[1::10]

    random.shuffle(train_set)
    random.shuffle(val_set)
    random.shuffle(test_set)

    x_train_set = [elem[0] for elem in train_set]
    y_train_set = [elem[1] for elem in train_set]
    x_validate_set = [elem[0] for elem in val_set]
    y_validate_set = [elem[1] for elem in val_set]
    x_test_set = [elem[0] for elem in test_set]
    y_test_set = [elem[1] for elem in test_set]

    x_train_set = np.array(x_train_set)
    x_test_set = np.array(x_test_set)
    x_validate_set = np.array(x_validate_set)

    y_train_set = np.array(y_train_set)
    y_test_set = np.array(y_test_set)
    y_validate_set = np.array(y_validate_set)

    return x_train_set, x_test_set, x_validate_set, y_train_set, y_test_set, y_validate_set

def load_dataset(path):
    dataset = []
    categories = {
        "Shetland_sheepdog": 0,
        "Papillon": 1,
        "Bernese_mountain_dog": 2,
        "Border_collie": 3,
        "Chow_chow": 4,
        "Pomeranian": 5,
        "Pug": 6,
        "Saluki": 7,
        "Samoyed": 8,
        "Siberian_husky": 9,
        "Other": 10
    }
    i = 0
    for directory in os.listdir(path):
        for image_path in os.listdir(path + "/" + directory):
            img = np.array(standardize((PIL.Image.open(path + "/" + directory + "/" + image_path)).convert('RGB')))
            dataset.append(np.array([img, categories[directory]]))
        i += 1
    return dataset


dataset = load_dataset("dogs")
normalized_dataset = normalize(dataset)
datagen = create_datagen_basic()


x_train_set, x_test_set, x_validate_set, y_train_set, y_test_set, y_validate_set = split_data(dataset)

x_train_norm = normalize(x_train_set)
x_test_norm = normalize(x_test_set)
x_validate_norm = normalize(x_validate_set)


#x_train_norm.extend(x_validate_norm)
#list(y_train_set).extend(y_validate_set)

model = Sequential([
  Rescaling(1./255, input_shape=(256, 256, 3)),
  layers.Conv2D(16, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(32, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(64, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Flatten(),
  layers.Dense(128, activation='relu'),
  layers.Dense(11)
])

model.compile(optimizer='adam',
              loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

print(x_train_set.shape)
print(y_train_set.shape)
epochs=10
history = model.fit(
  x=x_train_set,
  y=y_train_set,
  epochs=epochs
)
model.summary()

acc = history.history['accuracy']
loss = history.history['loss']

epochs_range = range(epochs)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.legend(loc='lower right')
plt.title('Training Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.legend(loc='upper right')
plt.title('Training Loss')
plt.show()

# mlp = MLPClassifier(verbose=0, random_state=0, max_iter=30, solver='sgd',
#                     learning_rate='constant', momentum=0, learning_rate_init=0.2)
# mlp = MLPClassifier(verbose=0, random_state=0, max_iter=30, solver='adam',
#                     learning_rate='invscaling', momentum=0, learning_rate_init=0.2)
# mlp = MLPClassifier(verbose=0, random_state=0, max_iter=30, solver='adam',
#                     learning_rate_init=0.01)
# nsamples, nx, ny, nz = x_train_set.shape
# x_train_set_reshaped = x_train_set.reshape((nsamples, nx*ny*nz))
# mlp.fit(x_train_set_reshaped, y_train_set)
# x_elem = np.array(x_train_set[0:1])
# shape, nx_t, ny_t, nz_t = x_elem.shape
# x_elem_reshaped = x_elem.reshape((1, nx_t*ny_t*nz_t))
# result = mlp.predict(x_train_set_reshaped)
# print(result)
# print(y_train_set[0])
# plt.plot(mlp.loss_curve_)
# plt.show()