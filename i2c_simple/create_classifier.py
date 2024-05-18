#creating decision tree classifier to classify motions as up, down left or right
import os 
import sys
import numpy as np 
import sklearn.tree as tree
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import pickle
from process_data1 import *
import sklearn.metrics as metrics
from sklearn import model_selection
from scipy.signal import butter, lfilter, freqz, iirnotch, filtfilt, firwin
import os

#returns a filtered signal filt
def filter_data(data_window_acc, data_window_gyro): #data frame snippet with acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z
    sampling_rate_acc = len(data_window_acc)/(data_window_acc['time'].astype(float).max() - data_window_acc['time'].astype(float).min())
    sampling_rate_gyro = len(data_window_gyro)/(data_window_gyro['time'].astype(float).max() - data_window_gyro['time'].astype(float).min())
    dt_gyro = data_window_gyro['time'].astype(float).to_numpy()[1] - data_window_gyro['time'].astype(float).to_numpy()[0]

    #I don't think we can use the buffered data approach because the movement is not consistent through the sample
    #buffer_size = 30

    #preprocesing the data window
    raw_x_acc = data_window_acc['x'].to_numpy()
    raw_y_acc = data_window_acc['y'].to_numpy()
    raw_z_acc = data_window_acc['z'].to_numpy()
    mag_acc = data_window_acc['magnitude'].to_numpy()
    raw_x_gyro = data_window_gyro['x'].to_numpy()
    raw_y_gyro = data_window_gyro['y'].to_numpy()
    raw_z_gyro = data_window_gyro['z'].to_numpy()

    #bandpass filter for acceleration signal:
    #cut off frequency --> this can be changed
    nyquist_rate = sampling_rate_acc/2.0
    
    f1 = 1/nyquist_rate
    f2 = 0.999

    #order is adjustable based on how much resolution we want in the data but must be an odd number
    order = int(0.3 * sampling_rate_acc)
    if order%2 == 0:
        order+=1
    a = np.array([1])
    #b = firwin(order,[f1, f2],pass_zero=False)
    b, a = butter(3, [f1, f2], btype='band')
    filt_acc = filtfilt(b, a, mag_acc)

    order = int(0.3 * sampling_rate_acc)
    if order%2 == 0:
        order+=1
    a = np.array([1])
    cutoff = 1.25/nyquist_rate
    #b = firwin(order,[f1, f2],pass_zero=False)
    b, a = butter(3, cutoff, btype='low')
    low_filt_acc_x = filtfilt(b, a, raw_x_acc)
    low_filt_acc_y = filtfilt(b, a, raw_y_acc)

    estimated_acc_angle = np.arctan(low_filt_acc_x/low_filt_acc_y)
    x_delta_gyro = raw_x_gyro * dt_gyro
    y_delta_gyro = raw_y_gyro * dt_gyro
    z_delta_gyro = raw_z_gyro * dt_gyro

    weight_gyro_angle = 0.98

    for i in range(len(data_window_gyro)):
        if i == 0:
            raw_x_gyro[i] = raw_x_gyro[i]
            raw_y_gyro[i] = raw_y_gyro[i]
            raw_z_gyro[i] = raw_z_gyro[i]
        else:
            raw_x_gyro[i] = weight_gyro_angle * (raw_x_gyro[i-1]+ x_delta_gyro[i]) + (1-weight_gyro_angle)*estimated_acc_angle[i-1]
            raw_y_gyro[i] = weight_gyro_angle*(raw_y_gyro[i-1]+ y_delta_gyro[i]) + (1-weight_gyro_angle)*estimated_acc_angle[i-1]
            raw_z_gyro[i] = weight_gyro_angle*(raw_z_gyro[i-1]+ z_delta_gyro[i]) + (1-weight_gyro_angle)*estimated_acc_angle[i-1]
    #mag_gyro = data_window_gyro['magnitude'].to_numpy()

    plt.plot(filt_acc, label='filt')
    plt.plot(mag_acc)
    plt.savefig('dummy.png')

    #bandpass filter for gryo:
    #cut off frequency --> this can be changed
    nyquist_rate = sampling_rate_gyro/2.0
    f1 = 0.5
    f2 = 0.75

    #order is adjustable based on how much resolution we want in the data but must be an odd number
    '''
    order = int(0.3 * sampling_rate_gryo)
    if order%2 == 0:
        order+=1
    a = np.array([1])
    b = firwin(order,[f1, f2],pass_zero=False)
    b, a = butter(3, [f1, f2], btype='band')
    filt_gyro= filtfilt(b, a, mag_acc)
    '''
    return filt_acc, raw_x_acc, raw_y_acc, raw_z_acc, mag_acc, raw_x_gyro, raw_y_gyro, raw_z_gyro


#returns a feature vector --> excluding gyro data for now for simplicity
def get_feats(filtered_sig, raw_x, raw_y, raw_z, mag, raw_x_gyro, raw_y_gyro, raw_z_gyro):
    feat_vector = []
    feat_names = []
    variance = np.var(filtered_sig)
    maximum = np.amax(filtered_sig)
    minimum = np.amin(filtered_sig)
    mean = np.mean(filtered_sig)
    feat_vector.append(variance)
    feat_vector.append(maximum)
    feat_vector.append(minimum)
    feat_vector.append(mean)
    feat_names.append('variance_acc')
    feat_names.append('max_acc')
    feat_names.append('min_acc')
    feat_names.append('mean_acc')

    x_variance = np.var(raw_x)
    x_maximum = np.amax(raw_x)
    x_minimum = np.amin(raw_x)
    x_mean = np.mean(raw_x)
    x_mean_gyro = np.mean(raw_x_gyro)
    prop_pos_gyro_x = sum(1 for i in raw_x_gyro if i >= 0 )/len(raw_x_gyro)
    feat_vector.append(x_variance)
    feat_vector.append(x_maximum)
    feat_vector.append(x_minimum)
    feat_vector.append(x_mean)
    feat_vector.append(prop_pos_gyro_x)
    feat_vector.append(x_mean_gyro)
    feat_names.append('variance_acc_x')
    feat_names.append('max_acc_x')
    feat_names.append('min_acc_x')
    feat_names.append('mean_acc_X')
    feat_names.append('prop_pos_gyro_X')
    feat_names.append('x_mean_gyro')

    y_variance = np.var(raw_y)
    y_maximum = np.amax(raw_y)
    y_minimum = np.amin(raw_y)
    y_mean = np.mean(raw_y)
    prop_pos_gyro_y = sum(1 for i in raw_y_gyro if i >= 0 )/len(raw_y_gyro)
    y_mean_gyro = np.mean(raw_y_gyro)
    feat_vector.append(y_variance)
    feat_vector.append(y_maximum)
    feat_vector.append(y_minimum)
    feat_vector.append(y_mean)
    feat_vector.append(prop_pos_gyro_y)
    feat_vector.append(y_mean_gyro)
    feat_names.append('variance_acc_y')
    feat_names.append('max_acc_y')
    feat_names.append('min_acc_y')
    feat_names.append('mean_acc_y')
    feat_names.append('prop_pos_gyro_y')
    feat_names.append('y_mean_gyro')

    z_variance = np.var(raw_z)
    z_maximum = np.amax(raw_z)
    z_minimum = np.amin(raw_z)
    z_mean = np.mean(raw_z)
    prop_pos_gyro_z = sum(1 for i in raw_z_gyro if i >= 0 )/len(raw_z_gyro)
    z_mean_gyro = np.mean(raw_z_gyro)
    feat_vector.append(z_variance)
    feat_vector.append(z_maximum)
    feat_vector.append(z_minimum)
    feat_vector.append(z_mean)
    feat_vector.append(prop_pos_gyro_z)
    feat_vector.append(z_mean_gyro)
    feat_names.append('variance_acc_z')
    feat_names.append('max_acc_z')
    feat_names.append('min_acc_z')
    feat_names.append('mean_acc_z')
    feat_names.append('prop_pos_gyro_z')
    feat_names.append('z_mean_gyro')

    #entropy calculation
    #derived from CS328:
    #Entropy = sum (p(x)log(p(x)))
    histogram = np.histogram(mag)[0]
    m_hist = []
    for h in histogram:
        if h ==0:
            m_hist.append(1)
        else:
            m_hist.append(h)
    entropy = sum(histogram*np.log10(m_hist))
    feat_vector.append(entropy)
    feat_names.append('entropy_acc')
    
    return np.array(feat_vector), np.array(feat_names)
   
def get_data():
    jerk_path = '/Users/anupamasitaraman/Downloads/i2cmod/i2c_simple/jerk_motion'
    slow_nod_path = '/Users/anupamasitaraman/Downloads/i2cmod/i2c_simple/slow_nod'

    jerk_list = os.listdir(jerk_path)
    nod_list = os.listdir(slow_nod_path)

    X = []
    Y = []
    #1 == Jerk, 0 == non-jerk
    for i in jerk_list:
        print(jerk_path+'/'+i)
        gyro, acc = process_dataframe(jerk_path+'/'+i)
        filt_acc, raw_x_acc, raw_y_acc, raw_z_acc, mag_acc, raw_x_gyro, raw_y_gyro, raw_z_gyro = filter_data(acc, gyro)
        feats, feat_names = get_feats(filt_acc, raw_x_acc, raw_y_acc, raw_z_acc, mag_acc, raw_x_gyro, raw_y_gyro, raw_z_gyro)
        X.append(feats)
        Y.append(1)
    for j in nod_list:
        gyro, acc = process_dataframe(slow_nod_path+'/'+j)
        filt_acc, raw_x_acc, raw_y_acc, raw_z_acc, mag_acc, raw_x_gyro, raw_y_gyro, raw_z_gyro = filter_data(acc, gyro)
        feats, feat_names = get_feats(filt_acc, raw_x_acc, raw_y_acc, raw_z_acc, mag_acc, raw_x_gyro, raw_y_gyro, raw_z_gyro)
        X.append(feats)
        Y.append(0)
    return np.array(X), np.array(Y)


    
def buildDecisionTree(x_data, y_data):
    predictions = []
    correct_labels = []
    cv = model_selection.KFold(n_splits=5, random_state=None, shuffle=True)
    for train, test in cv.split(x_data):
        #we can experiment wtih the hyperparameters
        t = tree.DecisionTreeClassifier(criterion="entropy", max_depth=10)
        t.fit(x_data[train], y_data[train])
        predicted = t.predict(x_data[test])
        predictions.append(predicted)
        correct_labels.append(y_data[test])
    
    #calcualte the stats for the model
    accuracies = []
    for i in range(len(predictions)):
        cm = metrics.confusion_matrix(correct_labels[i], predictions[i])
        print(cm)
        accuracies.append((cm[0][0]+cm[1][1])/(cm[0][0]+cm[1][0]+cm[1][1]+cm[0][1]))
    return t, accuracies

def buildRF(x_data, y_data):
    predictions = []
    correct_labels = []
    cv = model_selection.KFold(n_splits=5, random_state=None, shuffle=True)
    for train, test in cv.split(x_data):
        #we can experiment wtih the hyperparameters
        t = RandomForestClassifier(n_estimators = 5, max_depth=10)
        t.fit(x_data[train], y_data[train])
        predicted = t.predict(x_data[test])
        predictions.append(predicted)
        correct_labels.append(y_data[test])
    
    #calcualte the stats for the model
    accuracies = []
    for i in range(len(predictions)):
        cm = metrics.confusion_matrix(correct_labels[i], predictions[i])
        print(cm)
        accuracies.append((cm[0][0]+cm[1][1])/(cm[0][0]+cm[1][0]+cm[1][1]+cm[0][1]))
    return t, accuracies




#load data
#testing
#csv = '/Users/anupamasitaraman/Downloads/i2cmod/i2c_simple/slow_nod/slow_nod_1.csv'
#csv2 = '/Users/anupamasitaraman/Downloads/i2cmod/i2c_simple/jerk_motion/jerk_down_02.csv'
#gyro, acc = process_dataframe(csv)
#gyro2, acc2 = process_dataframe(csv2)
#filt_acc, raw_x_acc, raw_y_acc, raw_z_acc, mag_acc, filt_gyro, raw_x_gyro, raw_y_gyro, raw_z_gyro, mag_gyro = filter_data(acc2, gyro2)
#print(get_feats(filt_acc, raw_x_acc, raw_y_acc, raw_z_acc, mag_acc))

X, Y = get_data()
t, accuracies = buildDecisionTree(X, Y)
print(np.mean(accuracies))
rf, accuracies = buildRF(X,Y)
print(np.mean(accuracies))
filename = 'finalized_model.sav'
pickle.dump(t, open(filename, 'wb'))
filename = 'finalized_model_rf.sav'
pickle.dump(rf, open(filename, 'wb'))


r = tree.export_text(t) 
print(r)
#get sampling rate of the data:

#we are using the sliding window method here - we look at the signal in snippets
#set a window size: this is how much data we're going to be looking at a single time
window_size = 100
window_shift = 20

classes = ['jerk motion', 'slow nod']


#for each window, filter + detect if there is a movement
