import random
import PIL
from PIL import Image
import numpy as np
import os
import tensorflow as tf
from tensorflow import keras
from matplotlib import pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import gc

tf.config.list_physical_devices('GPU')

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


def augment_data(dataset):
    datagen = create_datagen_basic()
    generated_dataset = []
    for i in range(len(dataset)):
        gen = datagen.flow(np.stack(np.array(dataset)[i:i + 1, 0]), np.stack(np.array(dataset)[i:i + 1, 1]), batch_size=1)
        for j in range(16):
            generated_image = gen.next()[0][0]
            generated_dataset.append(np.array([generated_image, dataset[i][1]]))
    return generated_dataset


def create_datagen_basic():
    shift = 0.15

    datagen = ImageDataGenerator(
        featurewise_center=False,
        featurewise_std_normalization=False,
        rotation_range=20,
        width_shift_range=shift,
        height_shift_range=shift,
        horizontal_flip=True,
        zca_whitening=False,
        fill_mode='reflect',
        brightness_range=(0.6, 1.4)
    )

    return datagen

def standardize(image):
    size = (128, 128)
    ret = image.resize(size)
    return ret


def normalize(dataset):
    dataset_np = np.array(dataset)
    normalized_dataset = dataset_np / 255
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

    x_train_set = np.array(x_train_set, 'float16')
    x_test_set = np.array(x_test_set, 'float16')
    x_validate_set = np.array(x_validate_set, 'float16')

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
        if directory != "Other":
            for image_path in os.listdir(path + "/" + directory):
                img = np.array(standardize((PIL.Image.open(path + "/" + directory + "/" + image_path)).convert('RGB')), 'ubyte')
                dataset.append(np.array([img, categories[directory]]))
            i += 1
    return dataset


def load_other(path, number):
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
        if directory == "Other":
            for image_path in os.listdir(path + "/" + directory):
                img = np.array(standardize((PIL.Image.open(path + "/" + directory + "/" + image_path)).convert('RGB')),
                               'ubyte')
                dataset.append(np.array([img, categories[directory]]))
            i += 1
    random.shuffle(dataset)
    dataset = dataset[0:number]
    return dataset

dataset = load_dataset("dogs")
other = load_other("dogs", 16*194)
data_aug = augment_data(dataset)
data_aug.extend(other)

x_train_set, x_test_set, x_validate_set, y_train_set, y_test_set, y_validate_set = split_data(data_aug)

del data_aug
del dataset
gc.collect()
x_train_norm = normalize(x_train_set)
x_test_norm = normalize(x_test_set)
x_validate_norm = normalize(x_validate_set)

x_train_norm = np.append(x_train_norm, x_test_norm, 0)
y_train_set = np.append(y_train_set, y_test_set, 0)

# MODEL 1
model1 = keras.models.Sequential([
    keras.layers.Conv2D(filters=96, kernel_size=(11,11), strides=(4,4), activation='relu'),
    keras.layers.BatchNormalization(),
    keras.layers.MaxPool2D(pool_size=(3,3), strides=(2,2)),
    keras.layers.Conv2D(filters=256, kernel_size=(5,5), strides=(1,1), activation='relu', padding="same"),
    keras.layers.BatchNormalization(),
    keras.layers.MaxPool2D(pool_size=(3,3), strides=(2,2)),
    keras.layers.Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), activation='relu', padding="same"),
    keras.layers.BatchNormalization(),
    keras.layers.Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), activation='relu', padding="same"),
    keras.layers.BatchNormalization(),
    keras.layers.MaxPool2D(pool_size=(3,3), strides=(2,2)),
    keras.layers.Flatten(),
    keras.layers.Dense(4096, activation='relu'),
    keras.layers.Dense(4096, activation='relu'),
    keras.layers.Dense(11, activation='softmax')
])


# MODEL 2
# model1 = keras.models.Sequential([
#     keras.layers.Conv2D(filters=96, kernel_size=(11,11), strides=(4,4), activation='relu'),
#     keras.layers.BatchNormalization(),
#     keras.layers.MaxPool2D(pool_size=(2,2), strides=(2,2)),
#     keras.layers.Conv2D(filters=256, kernel_size=(5,5), strides=(1,1), activation='relu', padding="same"),
#     keras.layers.BatchNormalization(),
#     keras.layers.MaxPool2D(pool_size=(2,2), strides=(2,2)),
#     keras.layers.Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), activation='relu', padding="same"),
#     keras.layers.BatchNormalization(),
#     keras.layers.Conv2D(filters=1024, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding="same"),
#     keras.layers.BatchNormalization(),
#     keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2)),
#     keras.layers.Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), activation='relu', padding="same"),
#     keras.layers.BatchNormalization(),
#     keras.layers.MaxPool2D(pool_size=(2,2), strides=(2,2)),
#     keras.layers.Flatten(),
#     keras.layers.Dense(4096, activation='relu'),
#     keras.layers.Dropout(0.3),
#     keras.layers.Dense(4096, activation='relu'),
#     keras.layers.Dropout(0.3),
#     keras.layers.Dense(11, activation='softmax')
# ])

# MODEL 3
# model1 = keras.models.Sequential([
#     keras.layers.Conv2D(filters=96, kernel_size=(11,11), strides=(4,4), activation='relu'),
#     keras.layers.BatchNormalization(),
#     keras.layers.MaxPool2D(pool_size=(3,3), strides=(2,2)),
#     keras.layers.Conv2D(filters=256, kernel_size=(5,5), strides=(1,1), activation='relu', padding="same"),
#     keras.layers.BatchNormalization(),
#     keras.layers.MaxPool2D(pool_size=(3,3), strides=(2,2)),
#     keras.layers.Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), activation='relu', padding="same"),
#     keras.layers.BatchNormalization(),
#     keras.layers.Conv2D(filters=384, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding="same"),
#     keras.layers.BatchNormalization(),
#     keras.layers.Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), activation='relu', padding="same"),
#     keras.layers.BatchNormalization(),
#     keras.layers.MaxPool2D(pool_size=(3,3), strides=(2,2)),
#     keras.layers.Flatten(),
#     keras.layers.Dense(4096, activation='relu'),
#     keras.layers.Dropout(0.3),
#     keras.layers.Dense(4096, activation='relu'),
#     keras.layers.Dropout(0.3),
#     keras.layers.Dense(4096, activation='relu'),
#     keras.layers.Dropout(0.3),
#     keras.layers.Dense(4096, activation='relu'),
#     keras.layers.Dropout(0.1),
#     keras.layers.Dense(11, activation='softmax')
# ])


model1.compile(optimizer='adam',
              loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

print(x_train_set.shape)
print(y_train_set.shape)
epochs=50
del x_train_set
del x_validate_set
del x_test_set
del x_validate_norm
del y_validate_set
gc.collect()
history = model1.fit(
  x=x_train_norm,
  y=y_train_set,
  validation_data=(x_test_norm, y_test_set),
  epochs=epochs
)
model1.summary()

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(epochs)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training Loss')
plt.show()

del x_train_norm
del y_train_set
gc.collect()

# predictions = model1.predict(x_test_norm[0:10])
#
# for i in range(10):
#     score = tf.nn.softmax(predictions[i])
#     Image.fromarray(np.array(np.array(x_test_norm[i])*255, 'int')).show()
#     print(score)
#     print(list(categories.keys())[y_test_set[i]])
#     print(
#         "This image most likely belongs to {} with a {:.2f} percent confidence."
#         .format(list(categories.keys())[list(categories.values()).index(np.argmax(score))], 100 * np.max(score))
#     )
