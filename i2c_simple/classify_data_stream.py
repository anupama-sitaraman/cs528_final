import serial
import time
import csv
import pickle
import pandas as pd
from process_data1 import *
from create_classifier import *
import pygame

pygame.mixer.init()
pygame.mixer.music.load('/Users/anupamasitaraman/Downloads/i2cmod/i2c_simple/mixkit-game-show-wrong-answer-buzz-950.mp3')


port_name = '/dev/cu.usbserial-130'
f = open('/Users/anupamasitaraman/Downloads/i2cmod/i2c_simple/finalized_model_rf.sav', 'rb')
model = pickle.load(f)

#file_to_be_saved = 'right_41.csv'
s = serial.Serial(port_name, 115200)
s.flushInput()
obj = time.gmtime(0) 
epoch = time.asctime(obj)
buffer_time = []
buffer_measure = []
while True:
    time_sec = time.time() 
    line = s.readline()
    line = str(line)
    
    #print('new line')
    if 'acce_x' in line or 'gyro_x' in line:
        #print(line)
        buffer_time.append(time_sec)
        buffer_measure.append(line)
    #print(time_sec)
    if int(time_sec)%3 == 0:
        #print(('hh'))
        df = pd.DataFrame(dict({'time':buffer_time, 'measure':buffer_measure}))
        if len(df) <= 100:
            continue
        else:
            #print(df)
            gyro, acc = process_dataframe1(df)
            filt_acc, raw_x_acc, raw_y_acc, raw_z_acc, mag_acc, raw_x_gyro, raw_y_gyro, raw_z_gyro = filter_data(acc, gyro)
            feat_vector, feat_names = get_feats(filt_acc, raw_x_acc, raw_y_acc, raw_z_acc, mag_acc, raw_x_gyro, raw_y_gyro, raw_z_gyro )
            prediction = model.predict(feat_vector.reshape(1, -1))
            print('PREDICTION')
            #print(prediction)
            if prediction[0] == 1:
                print('WAKE UP!!!')
                pygame.mixer.music.play()
            else:
                print('keep crusin\' :D')
            buffer_time = []
            buffer_measure = []

        
    
