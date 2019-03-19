import numpy as np
import cv2
import pickle
import tensorflow_hub as hub
import time

from sklearn.preprocessing import LabelEncoder
import h5py
import tensorflow as tf
from tensorflow.keras.models import Sequential , load_model
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten, BatchNormalization

#Set up camera
print('Setting up imaging')
font = cv2.FONT_HERSHEY_SIMPLEX
cap = cv2.VideoCapture(0)
barchart_fulllength=300
barchart_width = 20
barchart_gap = 25
barchart_offset = 200
threshold=0.05

print('Loading model')
#model = load_model('./saved_models/model1/trained_model')
module = hub.Module('https://tfhub.dev/google/imagenet/mobilenet_v2_100_128/feature_vector/2', \
                    trainable=True)

IMAGE_SIZE = hub.get_expected_image_size(hub.Module('https://tfhub.dev/google/imagenet/mobilenet_v2_100_128/feature_vector/2'))
print(IMAGE_SIZE)

model = Sequential()
resize_layer = tf.keras.layers.Lambda(lambda x: tf.image.resize_images(x, (128,128)), input_shape = (100,100,3))
FeatureVector128 = tf.keras.layers.Lambda(module, input_shape = IMAGE_SIZE+[3])

model.add(resize_layer)
model.add(FeatureVector128)
model.add(BatchNormalization())
#model5.add(Dropout(0.125))
model.add(Dense(51, activation='softmax'))

model.load_weights('./saved_models/model5/weights.h5')

'''
model = Sequential()
model.add(Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=(100,100,3)))
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(3, 3)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(264, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(95, activation='softmax'))
'''

fileObject = open("./labelEncodingShort.pkl", "rb")
labelEncoder = pickle.load(fileObject)
fileObject.close()

print('Predicting...')
print('Press q to quit')
while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Our operations on the frame come here
    small_frame = cv2.resize(frame, (100,100))

    predictions = model.predict(np.expand_dims(small_frame/255, axis=0))[0]
    argmaxes = np.argsort(predictions)[-4:]
    labels = labelEncoder.inverse_transform(argmaxes)

    start_pixels= np.array([0,380])
    for i in range(4):
        probability = predictions[argmaxes[i]]
        if probability > threshold:
            cv2.putText(frame, str(labels[i]), tuple(start_pixels), font, 0.8, (0, 0, 0), 2, cv2.LINE_AA)
            chart_start = start_pixels+ np.array([barchart_offset,-barchart_width])
            length = int(probability*barchart_fulllength)
            chart_end = chart_start + np.array([length, barchart_width])  #(predictions[argmaxes[i]]*barchart_fulllength, barchart_width)
            cv2.rectangle(frame, tuple(chart_start), tuple(chart_end),  (110, 110, 0), cv2.FILLED)
        start_pixels = start_pixels +np.array([0, barchart_gap])

    # Display the resulting frame
    cv2.imshow('frame', frame)
    #time.sleep(0.2)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()