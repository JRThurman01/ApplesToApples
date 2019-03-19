#from tensorflow.keras.preprocessing.image import img_to_array
#from picamera import PiCamera
#from tensorflow.keras.models import load_model
#from imutils.video import VideoStream
import numpy as np
import imutils
import time
#import cv2

# initialize the total number of frames that *consecutively* contain
# santa along with threshold required to trigger the santa alarm
print('[INFO] setting thresholds')
TOTAL_CONSEC = 0
TOTAL_THRESH = 20
MODEL_PATH = './model.h5'

print("[INFO] loading model...")
#model = load_model(MODEL_PATH)

# initialize if the Fruit has been detected
FruitDetected = False

# initialize the video stream and allow the camera sensor to warm up
print("[INFO] starting video stream...")
#vs = VideoStream(usePiCamera=True).start()
time.sleep(2.0)

# loop over the frames from the video stream
while True:
    print('going')
    '''
	# grab the frame from the threaded video stream and resize it
	# to have a maximum width of 400 pixels
    frame = vs.read()
    frame = imutils.resize(frame, width=400)
    # prepare the image to be classified by our deep learning network
    image = cv2.resize(frame, (28, 28))
    image = image.astype("float") / 255.0
    #image = img_to_array(image)
    image = np.expand_dims(image, axis=0)

    # classify the input image and initialize the label and
    # probability of the prediction
    #predictions = model.predict(image)[0]
    prediction=[0,1]
    if predictions.max() > 0.5:
        label = predictions.argmax()
        TOTAL_CONSEC += 1

        # check to see if we should raise the santa alarm
        if TOTAL_CONSEC >= TOTAL_THRESH:
            # indicate that fruit has been found
            FruitDetected = True
            #Do something when fruit is detected

    else:
        TOTAL_CONSEC = 0
        FruitDetected = False

    # build the label and draw it on the frame
    label = "{}: {:.2f}%".format(label, proba * 100)
    frame = cv2.putText(frame, label, (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # show the output frame
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break

# do a bit of cleanup
print("[INFO] cleaning up...")
cv2.destroyAllWindows()
vs.stop()
    '''