# -*- coding: utf-8 -*-
"""
Created on Wed May 13 22:38:15 2020

@author: BT
"""

import numpy as np
from os import makedirs, path, listdir
from shutil import copyfile

np.random.seed(42)

from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D, Flatten, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator

reducePicDim = 256

#try:
#    CNN = load_model("dogVScat.h5")
#except:
trainDatagen = ImageDataGenerator(rotation_range=30, rescale=1./255, horizontal_flip=0.1)
trainGenerator = trainDatagen.flow_from_directory(
    directory='D:/DataScience/DeepLerning/Data/dog_vs_cat/train', #r"./dogs-vs-cats/train/",
    target_size=(reducePicDim, reducePicDim),
    color_mode="rgb", batch_size=64,
    class_mode="categorical", shuffle=True, seed=42)
    
CNN = Sequential()
CNN.add(Conv2D(32,(5,5),activation='relu',input_shape=(reducePicDim,reducePicDim,3)))
CNN.add(MaxPool2D(pool_size=(3, 3)))
CNN.add(BatchNormalization())
CNN.add(Conv2D(32,(5,5),activation='relu'))
CNN.add(MaxPool2D(pool_size=(3, 3)))
CNN.add(BatchNormalization())
CNN.add(Conv2D(64,(3,3),activation='relu'))
CNN.add(MaxPool2D(pool_size=(2, 2)))
CNN.add(BatchNormalization())
CNN.add(Conv2D(64,(3,3),activation='relu'))
CNN.add(Flatten())
CNN.add(Dense(100,activation='relu'))
CNN.add(Dense(50,activation='relu'))
CNN.add(Dense(2,activation='softmax'))
CNN.summary()
CNN.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])


CNN.fit_generator(generator=trainGenerator, epochs=10, verbose=True)
CNN.save("dogVScat.h5")


testDatagen = ImageDataGenerator(rescale=1./255)
testGenerator = testDatagen.flow_from_directory(
    directory='D:/DataScience/DeepLerning/Data/dog_vs_cat/test', #r"./dogs-vs-cats/test/",
    target_size=(reducePicDim, reducePicDim),
    color_mode="rgb", batch_size=8,
    class_mode="categorical", shuffle=False)
myloss, acc = CNN.evaluate_generator(testGenerator, steps=len(testGenerator), verbose=True)
print('Acc: %.3f' % (acc * 100.0))

testGenerator.reset()
yP = CNN.predict_generator(testGenerator, steps=len(testGenerator), verbose=True)
yPClass = np.argmax(yP,axis=1)

cats = np.sum(testGenerator.classes == 0)
dogs = np.sum(testGenerator.classes == 1)
catsAsDogs = np.sum( np.abs(yPClass[testGenerator.classes == 0] -0) )
dogsAsCats = np.sum( np.abs(yPClass[testGenerator.classes == 1] -1) )
confMatrix = np.array([[(cats-catsAsDogs)/cats, catsAsDogs/cats],
                       [dogsAsCats/dogs, (dogs-dogsAsCats)/dogs]])

print(confMatrix)