# -*- coding: utf-8 -*-
"""
Created on Sat Sep 15 12:17:18 2018

Hackathon: HackRice8
Team Red Raider
Team members: Mostofa Adib Shakib, Gautam Bakliwal, Alejandro Nevarez, Samuel Okei 

"""
import time
import small_camera_appcopy as sc
sc
import tensorflow as tf
import pandas as pd
import numpy as np
from numpy import genfromtxt
import cv2
import os
from keras.models import Sequential
from keras.layers import Conv2D, ZeroPadding2D, Activation, Input, concatenate
from keras.models import Model
from keras.layers.normalization import BatchNormalization
from keras.layers.pooling import MaxPooling2D, AveragePooling2D
from keras.layers.merge import Concatenate
from keras.layers.core import Lambda, Flatten, Dense
from keras.initializers import glorot_uniform
from keras.engine.topology import Layer
from keras import backend as K
K.set_image_data_format('channels_first')
from fr_utils import *
from inception_blocks_v2 import *
#import small_camera_app as sc

#import small_camera_app as sc
#%matplotlib inline     
#%load_ext autoreload
#%autoreload 2   

#Helps in outputting the whole numpy array instead of partial output when array called
np.set_printoptions(threshold=np.nan)

print("Imported all libraries")
#sc.run()
#Helps limit the memory usage by GPU
#from keras.backend.tensorflow_backend import set_session
#config = tf.ConfigProto()
#config.gpu_options.per_process_gpu_memory_fraction = 0.3
#set_session(tf.Session(config=config))
print("executed")
FR = FaceRecognition(input_shape=(3, 96, 96))
print("executed")
print("Total Params:", FR.count_params())
#FR.output_shape
#(None, 128)
def triplet_loss(y_true, y_pred, alpha = 0.2):
    
    anchor, positive, negative = y_pred[0], y_pred[1], y_pred[2]

    pos_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, positive)))#,axis= -1)
    #print(pos_dist.shape)
    neg_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, negative)))#,axis= -1)
    #print(neg_dist.shape)
    basic_loss = tf.add(tf.subtract(pos_dist, neg_dist), alpha)        
    loss = tf.reduce_sum(tf.maximum(basic_loss,0))  
    
    return loss
print("executed")
with tf.Session() as test:
    tf.set_random_seed(1)
    y_true = (None, None, None)
    y_pred = (tf.random_normal([3, 128], mean=6, stddev=0.1, seed = 1),
              tf.random_normal([3, 128], mean=1, stddev=1, seed = 1),
              tf.random_normal([3, 128], mean=3, stddev=4, seed = 1))
    loss = triplet_loss(y_true, y_pred)
    
    print("loss = " + str(loss.eval()))
print("executed testing the triplet loss function")
FR.compile(optimizer = 'adam', loss = triplet_loss, metrics = ['accuracy'])
load_weights_from_FaceNet(FR)
print("executed")


def verification(image_path, identity, database, model):
        
    encoding = img_to_encoding(image_path, model)
    
    dist = np.linalg.norm(encoding - database[identity])

    if dist<0.5:
        print("Identity Verified as: " + str(identity) + ", You have the access!")
        door_open = True
    else:
        print(str(identity) + " not Verified. Please add your name to database for the access.")
        door_open = False
    #can also return distance 
    return dist, door_open

def storing(imagepath):
    database = {}
    
    #img1 = cv2.imread(r"C:\Users\Gautam\Desktop\CS\Dog Cat classification\images\danielle.png", 1)
    #img = img1[...,::-1]
    #print("start...")
    #img = np.around(np.transpose(img, (2,0,1))/255.0, decimals=12)
    #print("start...")
    #x_train = np.array([img])
    #embedding = FR.predict_on_batch(x_train)
    #print("start...")
    #database["danielle"] = embedding
    database["pet1"] = img_to_encoding(imagepath, FR)
    return database["pet1"]

def execution(database):
    difference, result = verification("storage2/petpic0.jpg", "pet1", database, FR)
    return result

sc
time.sleep(90)
database = {}
database["pet1"] = storing(r"C:\Users\Gautam\Desktop\CS\Dog Cat classification\storage1\petpic0.jpg")
print(database["pet1"])
r=execution(database)
ser.write(r)
