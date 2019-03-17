from picamera import PiCamera
import datetime as dt
import numpy as np
import time
import tensorflow as tf


# Set up the camera
camera = PiCamera()
camera.resolution = (100, 100)
camera.framerate = 10

file_name =  './test_'+str(dt.datetime.now())+'.jpg'
#camera.capture(file_name)

for i in range(10):
    time.sleep(2)
    output = np.empty((128, 112, 3), dtype=np.uint8)
    camera.capture(output, 'rgb')
    print(output[:100,:100,:])


#close camera
camera.close()
