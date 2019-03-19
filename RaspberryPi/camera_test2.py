import io
import time
import picamera
import numpy as np

import tensorflow
from tensorflow.keras.model import load_model, load_weights

model=load_model('./model.h5')


with picamera.PiCamera() as camera:
    camera.resolution = (128, 112)
    #stream = io.BytesIO()
    stream = np.empty((100,100,3), dtype=np.uint8)
    for foo in camera.capture_continuous(stream, format='rgb'):
        # Truncate the stream to the current position (in case
        # prior iterations output a longer image)
        # output = np.empty((100, 100, 3), dtype=np.uint8)
        stream.truncate()
        stream.seek(0)
        print(stream[:100,:100,:])
#        if process(stream):
 #           break
