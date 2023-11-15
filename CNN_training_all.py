import numpy as np
import pandas as pd
from tensorflow import keras
from keras import Sequential
from keras.layers import Conv2D, MaxPool2D, Dense, Flatten, Dropout
from sklearn.model_selection import train_test_split
import cv2
from keras.preprocessing.image import ImageDataGenerator

import os
path = os.path.dirname(__file__)
cur_path = path + '/GTSRB_Dataset/archive/'

# Load train.csv
train_df = pd.read_csv(cur_path+'Train.csv')

# Extract the round signs 
round_signs = [0,1,2,3,4,5,7,8]
# train_df = train_df[(train_df['ClassId'].isin(round_signs))]

# # Add 'y_speed_limit_sign' column
# train_df['y_speed_limit_sign'] = train_df['ClassId'].apply(lambda x: 1 if x in range(0,8) else 0) #0 is wrong

# Add the speed limit numbers instead of 'ClassId'
# train_df.rename(columns={'ClassId':'Speed_limit'},inplace=True)

# If not a speed sign then assign 'ClassId' = 6
def assign_limits(x):
    if x not in round_signs:
        return 6
    else: 
        return x
    
train_df['ClassId'] = train_df['ClassId'].apply(assign_limits)

# Correct the path to images
train_df['Path'] = train_df['Path'].apply(lambda x: cur_path + x)

features = []
classes = []
labels = []

labels = train_df['ClassId']
for i, row in train_df.iterrows():
    try:
        # Load and resize image
        img = cv2.imread(row["Path"])
        img = img[row['Roi.Y1']:row['Roi.Y2'],row['Roi.X1']:row['Roi.X2']]
        img = cv2.resize(img, (40, 40))
        # img = img.reshape(1, 40, 40 ,3)
        # cv2.imshow('Original', img)
        # cv2.waitKey(0)
        features.append(img)
    except:
        print("Error loading image")

# for foldername in round_signs:
#     folder_path = os.path.join(cur_path,'Train',str(foldername))
#     images = os.listdir(folder_path)
#     for image in images:
#         try: 
#             img = cv2.imread(folder_path+'\\'+image)
#             img = cv2.resize(img, (40,40))
#             img = np.array(img)
#             features.append(img)
#             classes.append(foldername)
#         except:
#             print("Error loading image")

# for label in classes:
#     labels.append(assign_limits(label))

features = np.array(features)
labels = np.array(labels)

# Split into train and validation sets
X_train, X_val, y_train, y_val = train_test_split(features, labels, test_size=0.2, random_state=42)

#Converting the labels into one hot encoding
num_classes = len(np.unique(labels))
y_train = keras.utils.to_categorical(y_train,num_classes)
y_val = keras.utils.to_categorical(y_val, num_classes)

epochs = 30
batch_size = 32


cnn_model = keras.models.Sequential([
    keras.layers.Conv2D(filters=16, kernel_size=(3,3), activation='relu', input_shape=(40,40,3)),
    keras.layers.Conv2D(filters=32, kernel_size=(3,3), activation='relu'),
    keras.layers.MaxPool2D(pool_size=(2, 2)),
    keras.layers.BatchNormalization(axis=-1),
    
    keras.layers.Conv2D(filters=64, kernel_size=(3,3), activation='relu'),
    keras.layers.Conv2D(filters=128, kernel_size=(3,3), activation='relu'),
    keras.layers.MaxPool2D(pool_size=(2, 2)),
    keras.layers.BatchNormalization(axis=-1),
    
    keras.layers.Flatten(),
    keras.layers.Dense(512, activation='relu'),
    keras.layers.BatchNormalization(),
    keras.layers.Dropout(rate=0.5),
    
    keras.layers.Dense(num_classes, activation='softmax')
])

optimizer=keras.optimizers.Adam(learning_rate=0.001)
cnn_model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

aug = ImageDataGenerator(
    rotation_range=10,
    zoom_range=0.15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.15,
    horizontal_flip=False,
    vertical_flip=False,
    fill_mode="nearest")

cnn_history = cnn_model.fit(aug.flow(X_train, y_train, batch_size=batch_size), epochs=epochs, validation_data=(X_val, y_val))

cnn_model.save(path+'\CNN_model_all')