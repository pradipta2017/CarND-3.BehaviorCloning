import csv
import cv2
import numpy as np

csvpath = "data/pradipta/driving_log.csv"
imagepath = "data/pradipta/"
steering_correction = 0.25
samples = []

with open(csvpath) as  csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        samples.append(line)

def getImages(samples):
    images = []
    angles = []
    for line in samples:
        angle_dir = []

        center_filename = line[0]
        left_filename = line[1].strip(' ')
        right_filename = line[2].strip(' ')

        steering_center = float(line[3])
        angle_dir.append(steering_center)
        angle_dir.append(steering_center + steering_correction )
        angle_dir.append(steering_center - steering_correction)

        center_path = imagepath + center_filename
        left_path = imagepath + left_filename
        right_path = imagepath + right_filename
        
        center_image = cv2.cvtColor(cv2.imread(center_path),cv2.COLOR_BGR2RGB)
        left_image = cv2.cvtColor(cv2.imread(left_path),cv2.COLOR_BGR2RGB)
        right_image = cv2.cvtColor(cv2.imread(right_path),cv2.COLOR_BGR2RGB)

        images.append(center_image)
        images.append(left_image)
        images.append(right_image)

        angles.extend(angle_dir)

    return np.array(images), np.array(angles)



def flip_images(imgs,msmnts):
    x_imgs, x_mmnts = [],[]

    for image, measurement in zip(imgs,msmnts):
        x_imgs.append(image)
        x_mmnts.append(measurement)
        x_imgs.append(cv2.flip(image,1))
        x_mmnts.append(measurement*-1.0)
        
    return np.array(x_imgs), np.array(x_mmnts)



from keras.models import Sequential
from keras.layers import Cropping2D
from keras.layers import Flatten, Dense, Lambda, Dropout
from keras.layers.convolutional import Conv2D



def nvidia_model():
    nv_model = Sequential()
    nv_model.add(Lambda(lambda x: x/255 - 0.5, input_shape=X_train[0].shape))
    nv_model.add(Cropping2D(cropping=((50,20), (0,0)), input_shape=X_train[0].shape))
    nv_model.add(Conv2D(24,(5,5),subsample=(2,2), activation="relu"))
    nv_model.add(Conv2D(36,(5,5),subsample=(2,2), activation="relu"))
    nv_model.add(Conv2D(48,(5,5),subsample=(2,2), activation="relu"))
    nv_model.add(Conv2D(64,(3,3), activation="relu"))
    nv_model.add(Conv2D(64,(3,3), activation="relu"))
    nv_model.add(Flatten())
    nv_model.add(Dense(100))
    nv_model.add(Dropout(0.5))
    nv_model.add(Dense(50))
    nv_model.add(Dropout(0.5))
    nv_model.add(Dense(10))
    nv_model.add(Dense(1))

    return nv_model



X_train, y_train = getImages(samples)
X_train, y_train =  flip_images(X_train, y_train)

model = nvidia_model()
 

model.compile(loss="mse", optimizer="adam")
model.fit(X_train, y_train, validation_split=0.2, shuffle=True, epochs=5)
model.save("model.h5")
