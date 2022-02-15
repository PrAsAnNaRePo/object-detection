import tensorflow as tf
import numpy as np
import cv2
from collections import Counter
import os
import pandas as pd
from tensorflow.keras import Sequential, losses, layers, Model
from tensorflow import keras
from tensorflow.keras.applications import VGG16, resnet_v2

#phy = tf.config.list_physical_devices('GPU')
#tf.config.experimental.set_memory_growth(phy[0], True)


# HYPER parameters..
LIMIT = 6000
IMAGE_SIZE = (180, 180)
IMAGES = []
BBOX = []
LEARNING_RATE = 0.01
EPOCHS = 24

# data set loAded
df = pd.read_csv(r"D:\obj-dect\Self Driving Car.v3-fixed-small.tensorflow\export\_annotations.csv")

IMAGE_PATHS = [str(i) for i in df['filename'][0:LIMIT]]

for i in IMAGE_PATHS:
    img = cv2.imread(f"D:/obj-dect/Self Driving Car.v3-fixed-small.tensorflow/export/{i}")
    IMAGES.append(img)

print(f'{len(IMAGES)} images loaded...')

def resize_imgs(IMAGES):
    IMAGE = []
    for i in IMAGES:
        im = cv2.resize(i, IMAGE_SIZE)
        IMAGE.append(im)
    return IMAGE

# resizing images with 180x180
IMAGES = resize_imgs(IMAGES)


print(f'image size : {IMAGES[0].shape}')

for i in range(LIMIT):
    BBOX.append([df['xmin'][i], df['ymin'][i], df['xmax'][i], df['ymax'][i]])


def normalize_bounding_boxes(BBOX):
    bbox = []
    for i in BBOX:
        bbox.append(np.around((np.array(i) / 512) * 180))
    return bbox

# normalizing images should be done when when you adjest the image size
BBOX = normalize_bounding_boxes(BBOX)

LABELS = df['class'].values
LABELS = LABELS[0:6000]

def str_classToArray(LABELS):
    L = []# size = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    for i in LABELS:
        if str(i) == 'car':
            L.append([1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        if i == 'pedestrian':
            L.append([0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        if i == 'biker':
            L.append([0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0])
        if i == 'truck':
            L.append([0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0])
        if i == 'trafficLight-Red':
            L.append([0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0])
        if i == 'trafficLight':
            L.append([0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0])
        if i == 'trafficLight-Green':
            L.append([0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0])
        if i == 'trafficLight-RedLeft':
            L.append([0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0])
        if i == 'trafficLight-GreenLeft':
            L.append([0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0])
        if i == 'trafficLight-Yellow':
            L.append([0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0])
        if i == 'ttrafficLight-YellowLeft':
            L.append([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1])
    return L


LABELS = str_classToArray(LABELS)


x_train = IMAGES[0:int(len(IMAGES) * 0.80)]
x_train = np.array(x_train, dtype='float32')

x_test = IMAGES[int(len(IMAGES) * 0.80):]
x_test = np.array(x_test, dtype='float32')


y_train_LABELS = LABELS[0:int(len(IMAGES) * 0.80)]
y_train_LABELS =  np.array(y_train_LABELS)


y_test_LABELS = LABELS[int(len(IMAGES) * 0.8):]
y_test_LABELS =  np.array(y_test_LABELS)


y_train_bbox = BBOX[0:int(len(IMAGES) * 0.8)]
y_train_bbox = np.array(y_train_bbox)


y_test_bbox = BBOX[int(len(IMAGES) * 0.8):]
y_test_bbox = np.array(y_test_bbox)

print(x_train.shape)


base_model = resnet_v2.ResNet50V2(weights=None, include_top=False,input_tensor=keras.Input(shape=(180, 180, 3)))
# freeze all VGG layers so they will *not* be updated during the
# training process
base_model.trainable = False
# flatten the max-pooling output of VGG
x = base_model.output
flatten = layers.Flatten()(x)
# construct a fully-connected layer header to output the predicted
# bounding box coordinates
bboxHead = layers.Dense(128, activation="relu")(flatten)
bboxHead = layers.Dense(64, activation="relu")(bboxHead)
bboxHead = layers.Dense(32, activation="relu")(bboxHead)
output1 = layers.Dense(11, activation='softmax', name = 'labels')(bboxHead)
output2 = layers.Dense(4, activation='linear', name = 'bbox')(bboxHead)
# construct the model we will fine-tune for bounding box regression
model = keras.Model(inputs=base_model.input, outputs = [output1, output2])

model.compile(loss={
    'labels' : 'categorical_crossentropy',
    'bbox' : 'mse'
},
             optimizer = 'Adam',
             metrics=['accuracy'],
             )

y_train = {
    'labels':y_train_LABELS,
    'bbox' : y_train_bbox,
}
y_test = {
    'labels':y_test_LABELS,
    'bbox' : y_test_bbox,
}


model.fit(x_train, y_train, epochs=EPOCHS, validation_data=(x_test, y_test), batch_size=2)

model.save('resnet-v1')
