# -*- coding: utf-8 -*-
"""
Created on Sun May 17 13:25:59 2020

@author: BT
"""
import numpy as np
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import Flatten, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator

reducePicDim = 256


def computeConfusionMatrix(modell):
    testGenerator.reset()
    yP = modell.predict_generator(testGenerator, steps=len(testGenerator), verbose=True)
    yPClass = np.argmax(yP,axis=1)
    
    cats = np.sum(testGenerator.classes == 0)
    dogs = np.sum(testGenerator.classes == 1)
    catsAsDogs = np.sum( np.abs(yPClass[testGenerator.classes == 0] -0) )
    dogsAsCats = np.sum( np.abs(yPClass[testGenerator.classes == 1] -1) )
    confMatrix = np.array([[(cats-catsAsDogs)/cats * 100, catsAsDogs/cats * 100],
                           [dogsAsCats/dogs * 100, (dogs-dogsAsCats)/dogs * 100]])
    
    return confMatrix

'''
Es reichen hier wirklich 5 Epochen zum Trainieren. Es wird danach nicht mehr besser, manchmal
sogar schlechter. Und auch wenn der Rumpf nicht trainierbar ist generiert dieser amAusgang
32768 Merkmale. Das bedeutet unsere Klassifikator der darauf aufsetzt hat nun 3282052
Freiheitsgrade. Noch mal zum Vergleich, unser altes Modell was wir selbst gebaut haben hatte
nur 729252 Freiheitsgrade.

Das Ergebnisse ist eine nur verbessere Genauigkeit von 94.1 % gegenüber 89.4% 
bei unserem kleinen Netz (siehe 10_3).

Wenn Sie die konfusionmatrix mit der aus abschnitt 10_3 vergleichen, werden Sie bemerken, 
dass wir unseren ganzen Fortschritt den Katzen zu verdanken haben. Bei der Erkennung der Hunde 
haben wir uns nicht verbessert. 
'''
trainDatagen = ImageDataGenerator(rotation_range=30, rescale=1./255, horizontal_flip=0.1)
trainGenerator = trainDatagen.flow_from_directory(
    directory='D:/DataScience/DeepLerning/Data/dog_vs_cat/train', 
    target_size=(reducePicDim, reducePicDim), 
    color_mode="rgb", 
    batch_size=16, 
    class_mode="categorical", 
    shuffle=True, seed=42)

testDatagen = ImageDataGenerator(rescale=1./255)
testGenerator = testDatagen.flow_from_directory(
    directory='D:/DataScience/DeepLerning/Data/dog_vs_cat/test', #r"./dogs-vs-cats/test/",
    target_size=(reducePicDim, reducePicDim),
    color_mode="rgb", batch_size=8,
    class_mode="categorical", shuffle=False)

try:
    CNN = load_model("dogVScatVGGjustDense.h5")
except:
    stumpVGG16 = VGG16(weights='imagenet', include_top=False,
                       input_shape=(reducePicDim, reducePicDim, 3))
    stumpVGG16.trainable=False
    
flat = Flatten()(stumpVGG16.output)
x = Dense(100, activation='relu', name="dense1CatsVsDogs")(flat)
x = Dense(50, activation='relu', name="dense2CatsVsDogs")(x)
output = Dense(2, activation='softmax', name="softmaxCatsVsDogs")(x)

CNN = Model(stumpVGG16.inputs, output, name='VGGCatDog')

CNN.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
CNN.fit_generator(generator=trainGenerator, epochs=5, verbose=True)
CNN.save("dogVScatVGGjustDense.h5")

# Evaluate on test data
myloss, acc = CNN.evaluate_generator(testGenerator, steps=len(testGenerator), verbose=True)
print('Acc: %.3f' % (acc * 100.0))

# compute confusion matrix
confMatrix = computeConfusionMatrix(CNN)
print(confMatrix)

'''
Als wir das VGG16 Modell geladen haben, hatten wir sinnvolle 
Werte für die Featuregenerierung, wenn auch für eine andere Aufgabe. Für das Transfer Learning 
sind dabei zwei Ansätze generell sinnvoll
    - Das alte Modell hatte vorher eine allgemeinere Aufgabe ausgeführt und soll nun 
      spezialisiert werden
    - Das alte Modell hatte eine verwandte bzw. ähnliche Aufgabe ausgeführt und soll nun auf
      eine neue übertragen werden
Wir hatten den ersten Fall vorliegen. Wird ein Netz auf ImageNet trainiert so kann es bereits
Hunde und Katzen unterscheiden. Es erkennt aber auch Autos, Menschen etc. Dafür ist es
nicht so perfekt wenn es nur um Hunde und Katzen geht.

Abschließend hat man bereits sinnvolle Startwerte für die Filter und kann versuchen die mit 
seinen wenigen Daten zu spezialisieren.

Hierbei gibt es ein Problem was Sie nach den Diskussionen in Kapitel 7 und 8 sicherlich schon
erahnen. Wenn wir mit wenigen Daten nun ein Netz trainieren was mit vielen Freiheitsgraden
ausgestattet ist wird Overfitting verschärft zu einem Problem. Die Architektur war ja nicht für
unsere Mini-Datenbank vorgesehen. Außerdem kann es gut sein, dass wenn wir mit kleinen
Batches arbeiten der Optimierer uns zu schnell aus unsere sehr guten Startposition herausbewegt.
Die haben wir uns jedoch durch unseren Transferansatz so teuer erkämpft.

Was machen wir nun aus dieser Vorahnung, dass wir auf dünnem Eis laufen?
Wir nutzen unser Wissen über Optimierungsalgorithmen aus den vorhergehenden Kapitel 
und drosseln die Lernrate. Damit bewegen wir uns nicht mehr so schnell – also Vorsichtiger – 
und die Größe der Batches hat weniger Einfluss.
'''

try:
    CNN = load_model("dogVScatVGGPlus.h5")
except:
    from tensorflow.keras.optimizers import Adam
    CNN = load_model("dogVScatVGGjustDense.h5")
    slowADAM = Adam(learning_rate=0.0001)


for i in range(0,23): 
    CNN.layers[i].trainable=True
    
CNN.compile(optimizer=slowADAM,loss='categorical_crossentropy',metrics=['accuracy'])
CNN.summary()
CNN.fit_generator(generator=trainGenerator, epochs=5, verbose=True)
CNN.save("dogVScatVGGPlus.h5")

# Evaluate on test data
myloss, acc = CNN.evaluate_generator(testGenerator, steps=len(testGenerator), verbose=True)
print('Acc: %.3f' % (acc * 100.0))

'''
Da wir auf alle layer weiter trainieren Dauert der Optimierung wesentlich länger
weil die Optimierungsaufgabe nun viel mehr Freiheitsgrade hat.
Vermutlich über eine Stunde später erfahren Sie, dass wird es nun auf 97.1% Genauigkeit und
wir nun wieder viel besser in Hunden geworden sind.
Man könnte jetzt sagen, wenn dieses Netz meint es ist ein Hund, ist es auch ein Hund. Wenn
es Katze sagt sind leichte Zweifel erlaubt.
'''
# compute confusion matrix
confMatrix = computeConfusionMatrix(CNN)
print(confMatrix)