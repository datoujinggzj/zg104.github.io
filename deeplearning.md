---
layout: page
title: Deep Learning
subtitle: How deep is your love?
---

### How have I organized this page?
This page contains details about projects that I have worked on. In order to help readers get the crux of my work with a single look, I have organized each project using a self-designed template:

- The Project Overview
- The Links to the Project
- Main Processes of the Project
- Coding (If applicable)

## Animal Image Recognization in Deep Learning

![](image/cnn.png)

[Animal Image Recognization](https://github.com/zg104/Projects/blob/master/Deep%20Learning/cnn.py) by Zijing Gao.

__The project overview:__ Constructed the TensorFlow framework to establish a multiple hidden-layer CNN model for animal image recognition.

### **Data Preprocessing:** 
I collected a bunch of animal images, especially for dogs and cats, since they are pretty much the same and hard to distinguish. I always call my cat "年年" pronounced "Nian Nian" in English, and it looks like a cute dog sometimes. This name is created since I want him healthy and well. Most importantly, I want him [年年有余(鱼)!](https://chinesehacks.com/idioms/abundance-year-after-year/) 

I know exactly CNN is greedy, since it needs a ton of data to feed. I almost collected 10,000 more images in all, and I was planing to feed the image of "Nian Nian" as a single test sample into the model to see what would happen! <br/> I utilized ImageDataGenerator based on Keras in Python to rescale and transform the images for processing. 
 
You might want to know how cute he is!

![](image/nian.jpg)

### **Modeling:** 
As is known, CNN is different from the traditional deep neural networks, where extra processes are applied before we feed the data into neural networks. I created 3 by 3 feature detector matrix for filtering in each convoluntion layer. Then, I conducted paddling, pooling, and fully connected layers to squad and flatten the multidimensional data after several iterations of the previous steps.

### **Evaluation:** 
My poor laptop suffered from endless torture after several hours' training. I fed 20% of the original data as the test set to the model, and recieved recognization accuracy up to 85% after 25 epochs. That's cool since I have not tuned the parameters yet! The accuracy went up to around 92% after I added more convolution, pooling, and dropout layers, tuned the batch size, the number of epochs, and the optimizer. Finally, I fed 10 images of "Nian Nian" into my model, which gave me 100% accuracy back! Amazing! Sometimes, I should be proud of my photography skills.

### **Follow-on Work:** 
Actually, dogs and cats still have a lot of differences from head to toe. All of the images I collected are not limited to the facial images, and I would like to further explore the facial recognization of animals, evne emotion analysis of human beings! That is not a big deal for Google, but still challenging for a graduate student like me. I plan to work on millions of facial images based on CNN on the virtual machine to free my laptop. GCP is tasty and I will try my best to degest it.

### Coding

```python
###############################################
# Animal Image Recognization in Deep Learning #
###############################################

# import os
# os.chdir('C:\DeepLearning\Convolutional_Neural_Networks\dataset')

#### Part 1 - Building the CNN ####

# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

# Initialising the CNN
classifier = Sequential()

# Step 1 - Convolution

# nb_filter = the number of feature maps we create (nb feature detectors)
# (3,3) --> feature detectors shape
# input_shape --> shape we expect
classifier.add(Conv2D(32, (3, 3), input_shape = (64, 64, 3), activation = 'relu'))

# Step 2 - Pooling

# reduce the time we spent without losing features ( reduce the size of feature maps )
classifier.add(MaxPooling2D(pool_size = (2, 2))) # in general

# Adding a second convolutional layer
classifier.add(Conv2D(32, (3, 3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Step 3 - Flattening
# Q: why we don't lose spatial structure of feature maps
# A: because we use convolution step to extract the features through feature detector and
# we use max-pooling to extract the max of the feature map, which keeps the spatial features.
classifier.add(Flatten())

# Step 4 - Full connection
classifier.add(Dense(units = 128, activation = 'relu'))
classifier.add(Dense(units = 1, activation = 'sigmoid'))

# Compiling the CNN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

#### Part 2 - Fitting the CNN to the images ####

from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)

training_set = train_datagen.flow_from_directory('training_set',
                                                 target_size = (64, 64),
                                                 batch_size = 32,
                                                 class_mode = 'binary')


test_set = test_datagen.flow_from_directory('test_set',
                                            target_size = (64, 64), # if we want to increase the accuracy, we need to increase the size so that we get more info from the image.
                                            batch_size = 32,
                                            class_mode = 'binary')

classifier.fit_generator(training_set,
                         steps_per_epoch = 8000//32,
                         epochs = 25,
                         validation_data = test_set,
                         validation_steps = 2000//32)


import numpy as np
from keras.preprocessing import image
test_image = image.load_img('single_prediction/cat_or_dog_23.jpg', target_size = (64, 64))
test_image = image.img_to_array(test_image)

# add a new dimension
test_image = np.expand_dims(test_image, axis = 0)
result = classifier.predict(test_image)
training_set.class_indices
if result[0][0] == 1:
    prediction = 'dog'
else:
    prediction = 'cat'

print(prediction)

###########################################
# Improved Version After Parameter Tuning #
###########################################

from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator

# Image dimensions
img_width, img_height = 150, 150 # we use (64,64) previously. I think that is why the convergence is slow

"""
    Creates a CNN model
    p: Dropout rate
    input_shape: Shape of input 
"""


def create_model(p, input_shape=(32, 32, 3)):
    # Initialising the CNN
    model = Sequential()
    # Convolution + Pooling Layer
    model.add(Conv2D(32, (3, 3), padding='same', input_shape=input_shape, activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    # Convolution + Pooling Layer
    model.add(Conv2D(32, (3, 3), padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    # Convolution + Pooling Layer
    model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    # Convolution + Pooling Layer
    model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # we have 4 convolution + pooling layers in total compared to the original one (only 2)

    # Flattening
    model.add(Flatten())
    # Fully connection
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(p)) # avoid overfitting (train accuracy is high, but test accuracy is low)
    model.add(Dense(64, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(p / 2))
    model.add(Dense(1, activation='sigmoid'))

    # Compiling the CNN
    optimizer = Adam(lr=1e-3)
    metrics = ['accuracy']
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=metrics)
    return model


"""
    Fitting the CNN to the images.
"""


def run_training(bs=32, epochs=10):
    train_datagen = ImageDataGenerator(rescale=1. / 255,
                                       shear_range=0.2,
                                       zoom_range=0.2,
                                       horizontal_flip=True)
    test_datagen = ImageDataGenerator(rescale=1. / 255)

    training_set = train_datagen.flow_from_directory('training_set',
                                                     target_size=(img_width, img_height),
                                                     batch_size=bs,
                                                     class_mode='binary')

    test_set = test_datagen.flow_from_directory('test_set',
                                                target_size=(img_width, img_height),
                                                batch_size=bs,
                                                class_mode='binary')

    model = create_model(p=0.6, input_shape=(img_width, img_height, 3))
    model.fit_generator(training_set,
                        steps_per_epoch=8000 / bs,
                        epochs=epochs,
                        validation_data=test_set,
                        validation_steps=2000 / bs)


def main():
    run_training(bs=32, epochs=100)


""" Main """
if __name__ == "__main__":
    main()
```

