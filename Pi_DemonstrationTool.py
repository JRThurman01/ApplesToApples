
ISRASPBERRYPI=False

print('Loading dependencies')

if ISRASPBERRYPI:
    from picamera.array import PiRGBArray
    from picamera import PiCamera

import numpy as np
import cv2
import pickle
import tensorflow_hub as hub
import time
import pandas as pd
import glob
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Lambda, Dropout

#Set up global variables
IMAGEHEIGHT = 600
SCOREBOXWIDTH = 500
BARCHARTLENGTH = SCOREBOXWIDTH-50
BARCHARTTHICKNESS = 30
BARCHARTGAP = 25
BARCHARTOFFSET = 8
THRESHOLD=0.05
FONT = cv2.FONT_HERSHEY_SIMPLEX


def createModel(outputshape, nodes=264):
    featureVectorString = 'https://tfhub.dev/google/imagenet/mobilenet_v2_100_128/feature_vector/2'
    module = hub.Module(featureVectorString, trainable=False, name='featureVector')
    height, width = hub.get_expected_image_size(module)

    resize_layer = Lambda(lambda x: tf.image.resize_images(x, (height, width)), input_shape=(100,100,3))
    FeatureVector128 = Lambda(module, input_shape=[height, width, 3])

    # Create Model
    model = Sequential()
    model.add(resize_layer)
    model.add(FeatureVector128)
    model.add(Dropout(0.25))
    model.add(Dense(nodes, activation='relu'))
    model.add(Dropout(0.25))
    model.add(Dense(outputshape, activation='softmax'))

    # Compile model
    model.compile(optimizer='adam',
                       loss='sparse_categorical_crossentropy',
                       metrics=['accuracy'])
    return model

def loadLabelEncoder(file):
    fileObject = open(file, "rb")
    labelEncoder = pickle.load(fileObject)
    fileObject.close()
    return labelEncoder

def loadModel(modelNumber):
    print('Loading model weights')
    if modelNumber == 1:
        modelName = 'Model 1'
        size = 95
        model = createModel(size)
        model.load_weights('./saved_models/model3/weights.h5')
        return model, loadLabelEncoder("./labelEncoding.pkl"), modelName

    elif modelNumber == 2:
        modelName = 'Model 2'
        size = 52
        model = createModel(size)
        model.load_weights('./saved_models/model5/weights.h5')
        return model, loadLabelEncoder("./labelEncodingFlickr.pkl"), modelName

    elif modelNumber == 3:
        modelName = 'Model 3'
        size = 11
        model = createModel(size)
        model.load_weights('./saved_models/model7/weights.h5')
        return model, loadLabelEncoder("./labelEncodingFlickrSubset.pkl"), modelName

    elif modelNumber == 4:
        modelName = 'Model 4'
        size = 11
        model = createModel(size, nodes=128)
        model.load_weights('./saved_models/model8/weights.h5')
        return model, loadLabelEncoder("./labelEncodingFlickrSubset.pkl"), modelName

def drawProbabilities(predictions, frame, label):

    #resize frame
    height = IMAGEHEIGHT
    currentheight = frame.shape[0]  # keep original height
    ratio = height/currentheight
    width = int(frame.shape[1]*ratio)
    dim = (width, height)
    frame = cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)

    #Create the scores, add the model and dataset details:

    if liveMode:
        dataText = 'Streaming...'
    else:
        dataText=dataName

    score_frame = 200*np.ones((IMAGEHEIGHT,SCOREBOXWIDTH,3), np.uint8)
    cv2.putText(score_frame, 'Model : {}' .format(modelName), (20,30), FONT, 0.8, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(score_frame, 'Data source : {}'.format(dataText), (20, 60), FONT, 0.8, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(score_frame, 'Image label : {}'.format(label), (20, 90), FONT, 0.8, (0, 0, 0), 2, cv2.LINE_AA)

    #get 4 highest predictions and labels
    argmaxes = np.argsort(predictions)[-4:]
    labels = labelEncoder.inverse_transform(argmaxes)

    start_pixels = np.array([20, 150])
    for i in range(4):
        probability = predictions[argmaxes[-1-i]]
        probability = int(probability /0.05)*0.05 #rounding of the probability
        if probability > THRESHOLD:
            predictedLabel = labels[-1-i]
            if predictedLabel == label:
                colour = (40, 230, 40)
            else:
                colour = (20,20,220)

            text = '{}. {} ({}%)' .format(i+1, predictedLabel, round(probability*100,0))
            cv2.putText(score_frame ,text , tuple(start_pixels), FONT, 0.8, (0, 0, 0), 2, cv2.LINE_AA)
            chart_start = start_pixels + np.array([0, BARCHARTOFFSET])
            length = int(probability * BARCHARTLENGTH)
            chart_end = chart_start + np.array(
                [length, BARCHARTTHICKNESS])
            cv2.rectangle(score_frame , tuple(chart_start), tuple(chart_end), colour, cv2.FILLED)
        start_pixels = start_pixels + np.array([0, BARCHARTGAP+BARCHARTTHICKNESS+BARCHARTOFFSET])

    return np.hstack((score_frame, frame))

def getImage(liveMode):
    if liveMode:
        time.sleep(0.1)
        if ISRASPBERRYPI:
            camera.capture(rawCapture, format="bgr")
            frame = rawCapture.array
        else:
            ret, frame = cap.read()

        label = 'Streaming...'
        return frame, label
    else:
        time.sleep(3.0)
        file = np.random.choice(filelist, 1)[0]
        label = file.split('\\')[-2]
        frame = cv2.imread(file)
        return frame, label

def getFileImages(id=1):
    print('Retrieving sample images')
    #Returns a list of filepaths of different images
    if id==1:
        dataName='Fruit 360 data'
        url= './data/fruits/fruits-360/Training/*/*.jpg'
    elif id==2:
        dataName='Flickr data'
        url = './data/flickr/*/*.jpg'
    else:
        dataName='Flickr data subset'
        url = './data/demonstration/*/*.jpg'

    return glob.glob(url, recursive=True), dataName


#Set up camera and graphing tools
print('Setting up imaging')

if ISRASPBERRYPI:
    camera = PiCamera()
    rawCapture = PiRGBArray(camera)
    liveMode = True
else:
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 640)
    #Load the model, start with number 1.
    liveMode = False
    filelist, dataName = getFileImages(1)


model, labelEncoder, modelName = loadModel(1)

print('Predicting...')
print('Press q to quit, l to toggle live mode')
while(True):
    # Capture frame
    frame, label = getImage(liveMode)

    # make prediction based on 100x100 frame, show image on big_frame
    small_frame = cv2.resize(frame, (100,100))
    small_frame = cv2.cvtColor(small_frame, cv2.COLOR_RGB2BGR)

    #calculate predictions
    predictions = model.predict(np.expand_dims(small_frame/255, axis=0))[0]
    frame = drawProbabilities(predictions, frame, label)

    # Display the resulting frame
    cv2.imshow('frame', frame)
    waitkey = cv2.waitKey(1) & 0xFF
    if waitkey != 255:
        if waitkey  == ord('q'):
            break
        elif waitkey  == ord('l'):
            print('change mode')
            liveMode = not liveMode
        elif waitkey == ord('1'):
            print('change model')
            model, labelEncoder, modelName = loadModel(1)
        elif waitkey == ord('2'):
            print('change model')
            model, labelEncoder, modelName = loadModel(2)
        elif waitkey == ord('3'):
            print('change model')
            model, labelEncoder, modelName = loadModel(3)
        elif waitkey == ord('4'):
            print('change model')
            model, labelEncoder, modelName = loadModel(4)
        elif waitkey == ord('a'):
            print('change dataset')
            filelist, dataName = getFileImages(1)
        elif waitkey == ord('b'):
            print('change dataset')
            filelist, dataName = getFileImages(2)
        elif waitkey == ord('c'):
            print('change dataset')
            filelist, dataName = getFileImages(3)
        elif waitkey == ord(' '):
            pause= True
            while pause:
                time.sleep(3.0)
                waitkey = cv2.waitKey(1) & 0xFF
                if waitkey == ord(' '):
                    pause=False

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()


