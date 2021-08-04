import math                      # providing access to the mathematical functions defined by the C standard
import matplotlib.pyplot as plt  # plotting library
import scipy                     # scientific computnig and technical computing
import cv2                       # working with, mainly resizing, images
import numpy as np               # dealing with arrays
import glob                      # return a possibly-empty list of path names that match pathname
import os   
import numpy as np                     # dealing with directories
import pandas as pd              # providing data structures and data analysis tools
import tensorflow as tf       
import itertools
import random
from random import shuffle       # mixing up or currently ordered data that might lead our network astray in training.
from tqdm import tqdm            # a nice pretty percentage bar for tasks. Thanks to viewer Daniel Bühler for this suggestion

from PIL import Image
from scipy import ndimage
from pathlib import Path
from sklearn.metrics import classification_report, confusion_matrix
from sklearn import metrics
%matplotlib inline
np.random.seed(1)

from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Activation, Dropout, Flatten, Dense
from keras.callbacks import ModelCheckpoint, EarlyStopping


train_dir = ('/gdrive/My Drive/CNN/train_set/')
test_dir = ('/gdrive/My Drive/CNN/test_set/')

cols = ['Label','Condition','Train images', 'Test images']
label = pd.read_csv("/gdrive/My Drive/CNN/data_plant.csv", names=cols, skiprows=1)

label = label['Condition']
label

--------------------------------------------------------Data Augmentation---------------------------------------------------------------
LR = 1e-3
height=150
width=150
channels=3
seed=1337
num_classes = 2
epochs = 2
data_augmentation = True
num_predictions = 20

#Train generator
train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')

train_generator = train_datagen.flow_from_directory(train_dir, 
                                                    target_size=(height,width),
                                                    batch_size = 10,
                                                    shuffle=True,
                                                    class_mode='categorical')

#Test generator
test_datagen = ImageDataGenerator(rescale=1./255)
validation_generator = test_datagen.flow_from_directory(test_dir, 
                                                         target_size=(height,width),
                                                         batch_size = 10, 
                                                         shuffle=False,
                                                         class_mode='categorical')

train_num = train_generator.samples
validation_num = validation_generator.samples


---------------------------------------------------------Model Training----------------------------------------------------
model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=(150, 150, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3), padding='same'))
model.add(Activation('relu'))

model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.5))

model.add(Dense(num_classes))
model.add(Activation('softmax'))


model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['acc'])
model.summary()



#Model Architecture using SVG 
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot
SVG(model_to_dot(model, show_shapes=True, show_layer_names=True, rankdir='TB').create(prog='dot', format='svg'))

from keras.utils.vis_utils import plot_model
plot_model(model, to_file='cnn_modelarch_04072021.png.png')

#Saving the Weights
filepath=str(os.getcwd()+"/cnnmodel_srtp.h5f")
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
# = EarlyStopping(monitor='val_acc', patience=15)
callbacks_list = [checkpoint]#, stopper]



history = model.fit_generator(train_generator,
                    steps_per_epoch = train_num/batch_size,
                    epochs = 20,
                    validation_data = train_generator,
                    validation_steps= validation_num/batch_size ,
                    callbacks=callbacks_list,
                    verbose = 1)

---------------------------------------------------Model Evaluation-----------------------------------------------------------

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(acc) + 1)

plt.title('Training and validation accuracy')
plt.plot(epochs, acc, 'red', label='Training acc')
plt.plot(epochs, val_acc, 'blue', label='Validation acc')
plt.legend()

plt.figure()
plt.title('Training and validation loss')
plt.plot(epochs, loss, 'red', label='Training loss')
plt.plot(epochs, val_loss, 'blue', label='Validation loss')
plt.legend()
plt.show()

#Saving architecture to JSON
# serialize model to JSON
model_json = model.to_json()
with open("cnn_trainlogs_04072021.json", "w") as json_file:
    json_file.write(model_json)

# convert the history.history dict to a pandas DataFrame:     
hist_df = pd.DataFrame(history.history)

hist_csv_file = 'cnn_history_04072021.csv'
with open(hist_csv_file, mode='w') as f:
    hist_df.to_csv(f)

#Accuracy
scores = model.evaluate(train_generator, verbose=1)
print('Test loss:', scores[0])
print("Test Accuracy: %.2f%%" % (scores[1]*100))


#Confusion Matrix and Classification Report
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

Y_pred = model.predict_generator(validation_generator, nb_validation_samples // 
batch_size+1)
y_pred = np.argmax(Y_pred, axis=1)

print('Confusion Matrix')
print(confusion_matrix(validation_generator.classes, y_pred))

print('Classification Report')
target_names = ['Healthy', 'Diseased']
print(classification_report(validation_generator.classes, y_pred, target_names=target_names))

-----------------------OR------------------------------

test_steps_per_epoch = np.math.ceil(validation_generator.samples / validation_generator.batch_size)
predictions = model.predict_generator(validation_generator, steps=test_steps_per_epoch)
# Get most likely class
predicted_classes = np.argmax(predictions, axis=1)

true_classes = validation_generator.classes
class_labels = list(validation_generator.class_indices.keys())

report = metrics.classification_report(true_classes, predicted_classes, target_names=class_labels)
print(report) 