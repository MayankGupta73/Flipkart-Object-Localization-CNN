# -*- coding: utf-8 -*-
"""
Created on Mon Feb 11 20:08:34 2019

@author: Mayank Gupta
"""

# Assuming the training and test data has been separated into two folders
# Training images path - "/train_img"
# Test images path - "/test_img"

import numpy as np
import pandas as pd

#Keras functions
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.layers import (AveragePooling2D, Conv2D, Dense, Dropout, Flatten, Input, MaxPooling2D, ZeroPadding2D) 
from keras.models import Model, Sequential


#Inport training and test matrix
df = pd.read_csv('training.csv')

print(df)

#Defining the Keras model
      
input_shape = (240, 320, 3)

model = Sequential()
model.add(Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=input_shape))
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (5, 5), padding='same', activation='relu'))
model.add(Conv2D(64, (5, 5), activation='relu'))
model.add(MaxPooling2D(pool_size=(4, 4)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(4, activation='relu'))


print(model.summary())

#Training model on data

model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])
X_train = np.zeros((1000,120,160,3))
y_train = np.zeros((1000,4))

#Saving model architecture
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)


#Training model for 10 epochs
for i in range(10):
  print("Running for epoch",i+1)
  #Generate Train and test data
  for index, row in df.iterrows():
      if index = 13999 or (index%1000 == 0 and index != 0):
        print("Training for index",index)
        model.fit(X_train, y_train, epochs=1, batch_size=16)
        
        X_train = np.zeros((1000,120,160,3))
        y_train = np.zeros((1000,4))

      fileName = "train_img/"+row['image_name']
      if os.path.isfile(fileName):
        image = load_img(fileName,target_size=(120,160))
        image = img_to_array(image)

        X_train[index%1000,:,:,:] = image/255.
        y_train[index%1000,:] = (row['x1']/2, row['x2']/2, row['y1']/2, row['y2']/2)
        #print("Done for ",index)
  
  if (i+1)%2 == 0:
    # serialize weights to HDF5
    model.save_weights("model4-"+str(i+1)+".h5")
    print("Saved model to disk")
    

