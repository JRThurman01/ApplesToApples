import numpy as np
import pandas as pd
from os import listdir
#import matplotlib.image as mpimg
import cv2

def filelist_from_directory(directory):
    '''
    Given the directory, return all folder in the directory
    Note:
    Insider the directory, the folders are the labels of the data.
    In each folder there are 500 ish photos of that fruit
    There is a training and test data set with equivalent data structure
    
    '''
    # list all folders (note folder names are the labels)
    fruit_folders = [folder for folder in listdir(directory)]
    
    #Loop through each folder and each jpg file. Record the Fruit label, Directory and filename.
    fruits_labels=[]
    for fruit in fruit_folders:
        fruit_folder_string = directory+'/'+fruit
        fruitpictures = listdir(fruit_folder_string)
        for picture in fruitpictures:
            fruits_labels.append([fruit, fruit_folder_string+'/'+picture, picture])
    
    #Return the list to a dataframe
    return pd.DataFrame(fruits_labels, columns=['Fruit','Full filename','File'])

def create_dataset(filelist):
    '''
    From a list of all jpg file, append the pictures as an array
    
    '''
    # Takes the list created and bring back an array of 100x100x3 representing the picture
    photos = []
    for photo in filelist['Full filename']: #Column 1 is the directory 
        im = cv2.imread(photo)
        photos.append(im)

    X = np.array(photos)
    y= filelist['Fruit'].values
    
    return X, y

def get_data(directory, start_number=0, number_of_items=1000000000000000):
    filelist = filelist_from_directory(directory)[start_number: start_number+number_of_items]
    return create_dataset(filelist)
