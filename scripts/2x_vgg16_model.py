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
from keras.models import Sequential, Model
from keras.layers import Dense, Conv2D, MaxPool2D , Flatten

## Helps easily manipulate the images
from keras.preprocessing.image import ImageDataGenerator 

from matplotlib.image import imread
from keras.preprocessing import image

 
AUTOTUNE = tf.data.experimental.AUTOTUNE



"""
Satellite Data 
"""
#data_dir = tf.keras.utils.get_file('flower_photos','https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz', untar=True)

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



"""
Satellite View Data
"""
########################################################
########################################################
########################################################
sat_data_dir = '../data/satellite_imgs'
sat_data_dir = pathlib.Path(sat_data_dir)       
## Sets the images queried in an array
sat_all_images = list(sat_data_dir.glob('*/*'))
sat_all_images = [str(path) for path in sat_all_images if path is not '.git']
print(len(sat_all_images))
random.shuffle(sat_all_images)

## Seems like this is trying to search the labels within the given get_file --> get the parent and the name
## See https://docs.python.org/3/library/pathlib.html --> For loop over all the label names
## Look at https://docs.python.org/3/library/pathlib.html
sat_all_labels = []
for path in sat_all_images:
    #Odd for some reason there's a .git path that was caught in the folder?
    #print(pathlib.Path(path).parent.name)
    if pathlib.Path(path).parent.name in label_names:
        label = label_names[pathlib.Path(path).parent.name]
        sat_all_labels.append(label)
    else:
        print(path)
        print(pathlib.Path(path))
        print(pathlib.Path(path).parent.name)
        print(pathlib.Path(path).parent)

#all_labels=[label_names[pathlib.Path(path).parent.name] for path in all_images]
 
sat_data_size=len(sat_all_images)
sat_train_test_split=(int)(sat_data_size*0.1)
# Spliting up data into training sets
sat_x_train=sat_all_images[sat_train_test_split:]
print(len(sat_x_train))
sat_x_test=sat_all_images[:sat_train_test_split]
sat_y_train=sat_all_labels[sat_train_test_split:]
print(len(sat_y_train))
sat_y_test=sat_all_labels[:sat_train_test_split]



"""
Street View Data
"""
########################################################
########################################################
########################################################
street_data_dir = '../data/streetview_imgs'
street_data_dir = pathlib.Path(sat_data_dir)

street_all_images = list(street_data_dir.glob('*/*'))
street_all_images = [str(path) for path in street_all_images if path is not '.git']
print(len(street_all_images))
random.shuffle(street_all_images)

## Seems like this is trying to search the labels within the given get_file --> get the parent and the name
## See https://docs.python.org/3/library/pathlib.html --> For loop over all the label names
## Look at https://docs.python.org/3/library/pathlib.html
street_all_labels = []
for path in street_all_images:
    #Odd for some reason there's a .git path that was caught in the folder?
    #print(pathlib.Path(path).parent.name)
    if pathlib.Path(path).parent.name in label_names:
        label = label_names[pathlib.Path(path).parent.name]
        street_all_labels.append(label)
    else:
        print(path)
        print(pathlib.Path(path))
        print(pathlib.Path(path).parent.name)
        print(pathlib.Path(path).parent)

street_data_size =len(street_all_images)
print("The street data size is...")
print(street_data_size)
street_train_test_split=(int)(street_data_size*0.1)
# Spliting up data into training sets
street_x_train=street_all_images[street_train_test_split:]
print(len(street_x_train))
street_x_test=street_all_images[:street_train_test_split]
street_y_train=street_all_labels[street_train_test_split:]
print(len(street_y_train))
street_y_test=street_all_labels[:street_train_test_split]

########################################################
########################################################
########################################################
 
IMG_SIZE = 244 # Set the constant here (orig size is 640)
BATCH_SIZE = 256
 
def _parse_data(x):
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

  return image#,y
 
def _input_fn(x):
  ## Reference : https://www.tensorflow.org/api_docs/python/tf/data/Dataset	
  
  #According to website: "Creates a Dataset whose elements are slices of the given tensors."
  ds=tf.data.Dataset.from_tensor_slices((x))
  #Tuns it from an array to a map
  ds=ds.map(_parse_data)

  Not sure how to shuffle now
  ds=ds.shuffle(buffer_size=data_size) 

  ds = ds.repeat()
  # Takes consecutive pixels in image and groups them - not sure why
  ds = ds.batch(BATCH_SIZE)
  #Takes newer images as current images being processed --> efficiency basically
  ds = ds.prefetch(buffer_size=AUTOTUNE)
  return ds




"""
Satellite View Model
"""
########################################################
########################################################
########################################################
sat_train_ds=_input_fn(sat_x_train)
sat_validation_ds=_input_fn(sat_x_test)

## Load VGG16 from here
IMG_SHAPE = (IMG_SIZE, IMG_SIZE, 3)
## Current weights is imagenet, include_top = false means no classification layers
sat_VGG16_MODEL=tf.keras.applications.VGG16(input_shape=IMG_SHAPE, include_top=False, weights='imagenet')
sat_VGG16_MODEL._name = 'sat_vgg16'
# This means weights will not be updated in training
sat_VGG16_MODEL.trainable=False

sat_global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
# Dense function turns it to a single prediction for each image


#prediction_layer = tf.keras.layers.Dense(len(label_names))
sat_fc_1 = tf.keras.layers.Dense(512)

# Sequential Model - makes a stack of layers - https://keras.io/getting-started/sequential-model-guide/
sat_model = tf.keras.Sequential([sat_VGG16_MODEL, sat_global_average_layer, sat_fc_1])
sat_model.add(tf.keras.layers.Dense(512, input_shape=(1,), activation = 'relu'))
########################################################
########################################################
########################################################



"""
Streetview Model
"""
########################################################
########################################################
########################################################
street_train_ds=_input_fn(street_x_train)


street_validation_ds=_input_fn(street_x_test)
## Load VGG16 from here
IMG_SHAPE = (IMG_SIZE, IMG_SIZE, 3)
## Current weights is imagenet, include_top = false means no classification layers
street_VGG16_MODEL=tf.keras.applications.VGG16(input_shape=IMG_SHAPE, include_top=False, weights='imagenet')
street_VGG16_MODEL._name = 'street_vgg16'
# This means weights will not be updated in training
street_VGG16_MODEL.trainable=False

street_global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
# Dense function turns it to a single prediction for each image
#prediction_layer = tf.keras.layers.Dense(len(label_names))
street_fc_1 = tf.keras.layers.Dense(512)

# Sequential Model - makes a stack of layers - https://keras.io/getting-started/sequential-model-guide/
street_model = tf.keras.Sequential([street_VGG16_MODEL, street_global_average_layer, street_fc_1])
street_model.add(tf.keras.layers.Dense(512, input_shape=(512,), activation='relu'))
########################################################
########################################################
########################################################

"""
The combined model
"""
sat_street_merged = tf.keras.layers.concatenate([sat_model.output, street_model.output])
sat_street_merged = tf.keras.layers.Dense(512, activation = 'relu')(sat_street_merged)
sat_street_merged = tf.keras.layers.Dropout(.3)(sat_street_merged)
sat_street_merged = tf.keras.layers.Dense(256, activation = 'relu')(sat_street_merged)
sat_street_merged = tf.keras.layers.Dropout(.3)(sat_street_merged)
sat_street_merged = tf.keras.layers.Dense(128,activation = 'relu')(sat_street_merged)
# prediction layer
sat_street_merged = tf.keras.layers.Dense(3)(sat_street_merged)
sat_street_model = tf.keras.models.Model([sat_model.input, street_model.input], sat_street_merged)
sat_street_model.summary()

#See loss functions at https://www.tensorflow.org/api_docs/python/tf/keras/losses - would need to look this up


sat_street_model.compile(optimizer=tf.optimizers.Adam(), loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),  metrics=["accuracy"])
print(sat_train_ds)
print(street_train_ds)
print(sat_validation_ds)
print(street_validation_ds)

# train_data = tf.data.Dataset.zip((sat_train_ds,  street_train_ds))
#train_data_iterator = iter(train_data)
#print(train_data_iterator)

# test_data = tf.data.Dataset.zip((sat_validation_ds, street_validation_ds))
#test_data_iterator = iter(test_data)

history = sat_street_model.fit([sat_train_ds,street_train_ds], _input_fn(sat_y_train), epochs=100, steps_per_epoch=2, validation_steps=2, validation_data=([sat_validation_ds, street_validation_ds], _input_fn(sat_y_test)))
validation_steps = 20
loss0,accuracy0 = sat_street_model.evaluate(([sat_validation_ds, street_validation_ds], sat_y_test), steps = validation_steps)
 


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