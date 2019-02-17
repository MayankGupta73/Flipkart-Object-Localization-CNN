# -*- coding: utf-8 -*-
"""
Created on Mon Feb 11 20:22:19 2019

@author: Mayank Gupta
"""

#Images for testing are in "/test_img"

import pandas as pd
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.models import model_from_json

# Model reconstruction from JSON file
with open('model(2).json', 'r') as f:
    model = model_from_json(f.read())

# Load weights into the new model
model.load_weights('model-10(1).h5')

print(model.summary())

df = pd.read_csv('test.csv')

x1 = []
x2 = []
y1 = []
y2 = []
for index, row in df.iterrows():
    fileName = "../test_img/"+row['image_name']
    image = load_img(fileName,target_size=(240,320))
    image = img_to_array(image)
    y = model.predict(image.reshape((1,240,320,3))/255.)
    #(x1, x2, y1, y2) = model.predict(image.reshape((1,240,320,3)))
    x1.append(int(y[0][0] * 2))
    x2.append(int(y[0][1] * 2))
    y1.append(int(y[0][2] * 2))
    y2.append(int(y[0][3] * 2))
    
    if index%100 == 0:
        print("Done for index",index)

df['x1'] = x1
df['x2'] = x2
df['y1'] = y1
df['y2'] = y2

print(df)


df.to_csv("result2-10.csv", index = False)