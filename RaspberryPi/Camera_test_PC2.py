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

def drawProbabilities(predictions):
    argmaxes = np.argsort(predictions)[-4:]
    labels = labelEncoder.inverse_transform(argmaxes)

    start_pixels = np.array([0, 380])
    for i in range(4):
        probability = predictions[argmaxes[i]]
        if probability > threshold:
            cv2.putText(frame, str(labels[i]), tuple(start_pixels), font, 0.8, (0, 0, 0), 2, cv2.LINE_AA)
            chart_start = start_pixels + np.array([barchart_offset, -barchart_width])
            length = int(probability * barchart_fulllength)
            chart_end = chart_start + np.array(
                [length, barchart_width])  # (predictions[argmaxes[i]]*barchart_fulllength, barchart_width)
            cv2.rectangle(frame, tuple(chart_start), tuple(chart_end), (110, 110, 0), cv2.FILLED)
        start_pixels = start_pixels + np.array([0, barchart_gap])
def loadModel():
    module = hub.Module('https://tfhub.dev/google/imagenet/mobilenet_v2_100_128/feature_vector/2', trainable=True)
    model = Sequential()
    resize_layer = tf.keras.layers.Lambda(lambda x: tf.image.resize_images(x, (128, 128)), input_shape=(100, 100, 3))
    FeatureVector128 = tf.keras.layers.Lambda(module, input_shape=(128, 128, 3))
    model.add(resize_layer)
    model.add(FeatureVector128)
    model.add(BatchNormalization())
    model.add(Dense(51, activation='softmax'))
    model.load_weights('./saved_models/model5/weights.h5')
    return model
def loadLabelEncoder():
    fileObject = open("./labelEncodingShort.pkl", "rb")
    labelEncoder = pickle.load(fileObject)
    fileObject.close()
    return labelEncoder

#Set up camera and graphing tools
print('Setting up imaging')
cap = cv2.VideoCapture(0)
barchart_fulllength=300
barchart_width = 20
barchart_gap = 25
barchart_offset = 200
threshold=0.05
font = cv2.FONT_HERSHEY_SIMPLEX

#Load the model
print('Loading model')
model = loadModel()
labelEncoder = loadLabelEncoder()

print('Predicting...')
print('Press q to quit')
while(True):
    # Capture frame
    ret, frame = cap.read()
    #frame = cv2.imread('./data/InstagramData/Apple/30br___BLNeW5uFxa7___.jpg')
    print(type(frame))

    # make prediction
    small_frame = cv2.resize(frame, (100,100))
    predictions = model.predict(np.expand_dims(small_frame/255, axis=0))[0]
    drawProbabilities(predictions)

    # Display the resulting frame
    cv2.imshow('frame', frame)
    time.sleep(1)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()


