"""
	The code is derived from 
	https://androidkt.com/how-to-use-vgg-model-in-tensorflow-keras/
	and 
	https://towardsdatascience.com/step-by-step-vgg16-implementation-in-keras-for-beginners-a833c686ae6c -->  does it step by step
"""
from tqdm import tqdm
from numpy.random import randn

import pathlib
import random
import keras,os

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
 
import tensorflow as tf
import numpy as np

#Keras import commands are from another website --> might not be necessary
import keras,os
#Sequential means that layers will be made in sequence- https://towardsdatascience.com/step-by-step-vgg16-implementation-in-keras-for-beginners-a833c686ae6c
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPool2D , Flatten
## Helps easily manipulate the images
from keras.preprocessing.image import ImageDataGenerator 

from matplotlib.image import imread
from keras.preprocessing import image

 
AUTOTUNE = tf.data.experimental.AUTOTUNE

data_dir = '../data/streetview_imgs'
#data_dir = tf.keras.utils.get_file('flower_photos','https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz', untar=True)
data_dir = pathlib.Path(data_dir)
print(data_dir)
#taken from https://realpython.com/working-with-files-in-python/
# and https://www.guru99.com/reading-and-writing-files-in-python.html
#with open == open("File Name", "r") as f:
#data_file = f.read();

#Only need this if taking the file from URL --> https://www.tensorflow.org/api_docs/python/tf/keras/utils/get_file
#data_dir = tf.keras.utils.get_file('put file name','put URL here', untar=True) 
##Import images as 640 * 640
#https://docs.python.org/3/library/pathlib.html
#data_file = pathlib.Path(data_file)

## This is just an example of a classification that we can set up but must correspond to what's in the image file

label_names={'bin1': 2, 'bin2': 0, 'bin3': 1} # bin1 --> cluster 0, bin2 --> cluster1, bin3 --> cluster 2
label_key=[2,0,1]

           
## Sets the images queried in an array
all_images = list(data_dir.glob('*/*'))
all_images = [str(path) for path in all_images if path is not '.git']
print(len(all_images))
random.shuffle(all_images)

## Seems like this is trying to search the labels within the given get_file --> get the parent and the name
## See https://docs.python.org/3/library/pathlib.html --> For loop over all the label names
## Look at https://docs.python.org/3/library/pathlib.html
all_labels = []
for path in all_images:
    #Odd for some reason there's a .git path that was caught in the folder?
    print(pathlib.Path(path).parent.name)
    if pathlib.Path(path).parent.name in label_names:
        label = label_names[pathlib.Path(path).parent.name]
        all_labels.append(label)
    else:
        print(path)
        print(pathlib.Path(path))
        print(pathlib.Path(path).parent.name)
        print(pathlib.Path(path).parent)




#all_labels=[label_names[pathlib.Path(path).parent.name] for path in all_images]
 
data_size=len(all_images)
 
train_test_split=(int)(data_size*0.1)
 
# Spliting up data into training sets
x_train=all_images[train_test_split:]
print(len(x_train))
x_test=all_images[:train_test_split]
 
y_train=all_labels[train_test_split:]
print(len(y_train))
y_test=all_labels[:train_test_split]
 
IMG_SIZE = 244 # Set the constant here (orig size is 640)
 
BATCH_SIZE = 256
 
def _parse_data(x,y):
  ##This processes the image data 
  image = tf.io.read_file(x)
  image = tf.image.decode_jpeg(image, channels=3)
  image = tf.cast(image, tf.float32)
  image = (image/127.5) - 1
  image = tf.image.resize(image, (IMG_SIZE, IMG_SIZE))
  ## Resize function has a method to choose the resizeing method
  ## Anti-aliasing, according to Google, is mixing the colors together --> not useful for us
  ## Resize method defaults to bilinear --> see https://en.wikipedia.org/wiki/Image_scaling
  ## https://www.tensorflow.org/api_docs/python/tf/image/resize

  return image,y
 
def _input_fn(x,y):
  ## Reference : https://www.tensorflow.org/api_docs/python/tf/data/Dataset	
  
  #According to website: "Creates a Dataset whose elements are slices of the given tensors."
  ds=tf.data.Dataset.from_tensor_slices((x,y))
  #Tuns it from an array to a map
  ds=ds.map(_parse_data)
  ds=ds.shuffle(buffer_size=data_size)

  ds = ds.repeat()
  # Takes consecutive pixels in image and groups them - not sure why
  ds = ds.batch(BATCH_SIZE)
  # Takes newer images as current images being processed --> efficiency basically
  ds = ds.prefetch(buffer_size=AUTOTUNE)
  
  return ds
  
train_ds=_input_fn(x_train,y_train)
validation_ds=_input_fn(x_test,y_test)

## Load VGG16 from here
IMG_SHAPE = (IMG_SIZE, IMG_SIZE, 3)
## Current weights is imagenet, include_top = false means no classification layers
VGG16_MODEL=tf.keras.applications.VGG16(input_shape=IMG_SHAPE, include_top=False, weights='imagenet')

# This means weights will not be updated in training
VGG16_MODEL.trainable=False

global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
# Dense function turns it to a single prediction for each image
prediction_layer = tf.keras.layers.Dense(len(label_names),activation='softmax')

# Sequential Model - makes a stack of layers - https://keras.io/getting-started/sequential-model-guide/
model = tf.keras.Sequential([VGG16_MODEL, global_average_layer, prediction_layer])

#See loss functions at https://www.tensorflow.org/api_docs/python/tf/keras/losses - would need to look this up
model.compile(optimizer=tf.optimizers.Adam(), loss=tf.keras.losses.SparseCategoricalCrossentropy(),  metrics=["accuracy"])

history = model.fit(train_ds, epochs=100, steps_per_epoch=2, validation_steps=2, validation_data=validation_ds)

validation_steps = 20
 
loss0,accuracy0 = model.evaluate(validation_ds, steps = validation_steps)
 
print("loss: {:.2f}".format(loss0))
print("accuracy: {:.2f}".format(accuracy0))

# Accuracy Plot
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

#Loss Function Plot
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
