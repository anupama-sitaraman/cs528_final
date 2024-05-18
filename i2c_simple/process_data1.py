import numpy as np 
import pandas as pd 
import scipy.signal as sp
import scipy.fft  as fft
import matplotlib.pyplot as plt
import re

def plot_line_acceleration(data, name):
    plt.plot(data['x'].to_numpy(), label='x axis')
    plt.plot(data['y'].to_numpy(), label='y axis')
    plt.plot(data['z'].to_numpy(), label='z axis')
    plt.plot(data['magnitude'].to_numpy(), label='overall magnitude')
    plt.title('acceleration plot of ' + name)
    plt.xlabel('time (s)')
    plt.legend()
    plt.savefig('acceleration_line_plot_'+name+'.png')
    
    plt.show()
def plot_line_gyro(data, name):
    plt.plot(data['x'].to_numpy(), label='x axis')
    plt.plot(data['y'].to_numpy(), label='y axis')
    plt.plot(data['z'].to_numpy(), label='z axis')
    #plt.plot(data['magnitude'].to_numpy(), label='overall magnitude')
    plt.title('gyroscope data of ' + name)
    plt.xlabel('time (s)')
    plt.legend()
    plt.savefig('gyroscope_line_plot_'+name+'.png')
    
    plt.show()
    



def plot_line_x_y_acceleration(data1, data2, name1, name2):
    plt.plot(data1['magnitude'].to_numpy(), label=name1 +' data')
    plt.plot(data2['magnitude'].to_numpy(), label=name2 +' data')
    plt.title('acceleration plot of ' + name1 +' and ' + name2)
    plt.xlabel('time (s)')
    plt.legend()
    plt.show()
    plt.savefig('acceleration_line_plot_'+name1+name2+'.png')

def plot_spec(data, name, isGyro):
    sampling_freq = 1/((data['time'].max() - data['time'].min())/len(data))
    nfft = 140
    noverlap = 50
    print(name)

    spec, freq, t, im = plt.specgram(data['x'].to_numpy(), Fs=sampling_freq, NFFT=nfft, scale='linear')
    plt.title(name +' x axis spectrogram')
    plt.ylabel('Frequncy (Hz)')
    plt.xlabel('time (seconds)')
    if isGyro == True:
        plt.colorbar(im, label=' angular velocity (deg/sec)' )
    else: 
        plt.colorbar(im, label =' acceleration (g)')
    plt.savefig('./'+name+'_x.png')
    plt.clf()

    spec1, freq1, t1, im1 = plt.specgram(data['y'].to_numpy(), Fs=sampling_freq, NFFT=nfft, scale='linear')
    plt.title(name +' y axis spectrogram')
    plt.ylabel('Frequncy (Hz)')
    plt.xlabel('time (seconds)')
    if isGyro == True:
        plt.colorbar(im1, label=' angular velocity (deg/sec)' )
    else: 
        plt.colorbar(im1, label =' acceleration (g)')
    plt.savefig('./'+name+'_y.png')
    plt.clf()

    spec2, freq2, t2, im2 = plt.specgram(data['z'].to_numpy(), Fs=sampling_freq, NFFT=nfft, scale='linear')
    plt.title(name +' z axis spectrogram')
    plt.ylabel('Frequncy (Hz)')
    plt.xlabel('time (seconds)')
    if isGyro == True:
        plt.colorbar(im2, label=' angular velocity (deg/sec)' )
    else: 
        plt.colorbar(im2, label =' acceleration (g)')
    plt.savefig('./'+name+'_z.png')
    plt.clf()

def process_dataframe1(df):
    
    df['measure'] = df['measure'].astype(str)
    print(df)
    #m= measure.str.re.findall(r'[-+]?\d+')
    #df = df[m]
    #print(df)
    fail_filt = df['measure'].str.contains('Failed')
    df = df[~fail_filt]
    df = df[df['measure'].str.contains('_y')]
    
    df['measure'] = df['measure'].str.replace('\\', '').str.replace('rn', '').str.replace(',', "").str.replace('\'', "")
    df = df.join(df['measure'].str.split('test:', expand=True))
    df = pd.DataFrame({'time': df['time'], 'one':df[1]})
    df = df.join(df['one'].str.split(':', expand=True))
    
    df['x'] = df[1].str.split(' ').str[0]
    df['y'] = df[2].str.split(' ').str[0]
    df['z'] = df[3]
    #df=df.iloc[3:]
    df['x'] = df['x'].astype(float)
    df['y'] = df['y'].astype(float)
    df['z'] = df['z'].astype(float)
    df_gyro = df[df[0] == ' gyro_x' ]
    df_acc = df[df[0] ==' acce_x']
    df_acc['magnitude'] = np.sqrt(df_acc['x']**2 + df_acc['y']**2 + df_acc['z']**2)
    df_gyro['magnitude'] = np.sqrt(df_gyro['x']**2 + df_gyro['y']**2 + df_gyro['z']**2)
    return df_gyro, df_acc
def process_dataframe(csv):
    print('here')
    df = pd.DataFrame(pd.read_csv(csv))
    df['measure'] = df['measure'].astype(str)
    df['measure'] = df['measure'].str.replace('\\', '').str.replace('rn', '').str.replace(',', "").str.replace('\'', "")
    #print(df)
    df = df.join(df['measure'].str.split('test:', expand=True))
    print(df)
    df = pd.DataFrame({'time': df['time'], 'one':df[1]})
    #df = df.drop(columns=['measure', 0])
    #df = df.rename(columns={1:'one'})
    #df = df.dropna(subset=['one'], inplace=True)
   
    #df[1] = df[1].astype(str)
    #print(df)
    df = df.join(df['one'].str.split(':', expand=True))
    df['x'] = df[1].str.split(' ').str[0]
    df['y'] = df[2].str.split(' ').str[0]
    df['z'] = df[3]
    df=df.iloc[3:]
    #print(df)
    
    '''
    df[['z', 'junk1', 'junk']] = df[4].str.split("\\", expand=True)
    df = df.drop(columns=['junk1', 'junk', 'measure', 4])
    df[['y', 'junk1']] = df[3].str.split(",", expand=True)
    df[['x', 'junk2']] = df[2].str.split(",", expand=True)
    df = df.drop(columns=['junk1', 'junk2', 2, 3, 0])
    print(df)
    df['x'] = df['x'].str.replace('\\', '')
    df['x'] = df['x'].str.replace('rn\'', '')
    df['y'] = df['y'].str.replace('\\', '')
    df['y'] = df['y'].str.replace('rn\'', '')
    df['z'] = df['z'].str.replace('\\', '')
    df['z'] = df['z'].str.replace('rn\'', '')
    df['x'] = df['x'].astype(float)
    df['y'] = df['y'].astype(float)
    df['z'] = df['z'].astype(float)
    #print(df[4][0].split('\\'))
    
    df = df.dropna()
    '''
    df['x'] = df['x'].astype(float)
    df['y'] = df['y'].astype(float)
    df['z'] = df['z'].astype(float)
    df_gyro = df[df[0] == ' gyro_x' ]
    df_acc = df[df[0] ==' acce_x']

    df_acc['magnitude'] = np.sqrt(df_acc['x']**2 + df_acc['y']**2 + df_acc['z']**2)
    #df_gyro['magnitude'] = df_gyro['x']+ df_gyro['y']+ df_gyro['z']
    #print(df)
    return df_gyro, df_acc

def generate_spects():  

    up_gyro, up_acc = process_dataframe('./up1.csv')
    print(up_gyro['x'])
    down_gyro, down_acc = process_dataframe('./down1.csv')
    left_gyro, left_acc = process_dataframe('./left1.csv')
    right_gyro, right_acc = process_dataframe('./right1.csv')
    
    plot_spec(up_gyro, 'up gyroscope hand motion', True)
    plot_spec(down_gyro, 'down gyroscope hand motion', True)
    plot_spec(left_gyro, 'left gyroscope hand motion', True)
    plot_spec(right_gyro, 'right gyroscope hand motion', True)

    plot_spec(up_acc, 'up accelerometer hand motion', False)
    plot_spec(down_acc, 'down accelerometer hand motion', False)
    plot_spec(left_acc, 'left accelerometer hand motion', False)
    plot_spec(right_acc, 'right accelerometer hand motion', False)
    
    up_gyro, up_acc = process_dataframe('./up2.csv')
    down_gyro, down_acc = process_dataframe('./down2.csv')
    left_gyro, left_acc = process_dataframe('./left2.csv')
    right_gyro, right_acc = process_dataframe('./right2.csv')
    
    plot_spec(up_gyro, 'up gyroscope wrist motion', True)
    plot_spec(down_gyro, 'down gyroscope wrist motion', True)
    plot_spec(left_gyro, 'left gyroscope wrist motion', True)
    plot_spec(right_gyro, 'right gyroscope wrist motion', True)

    plot_spec(up_acc, 'up accelerometer wrist motion', False)
    plot_spec(down_acc, 'down accelerometer wrist motion', False)
    plot_spec(left_acc, 'left accelerometer wrist motion', False)
    plot_spec(right_acc, 'right accelerometer wrist motion', False)
    
#generate_spects()

def hw_responses():
    print('here')
    data_n1 = 'left_41'
    data_n2 = 'left_39'
    data_name1 = 'left_15.csv'
    data_name2 = 'left_07.csv'
    data1_gyro, data1_acc = process_dataframe(data_n1+'.csv')
    data2_gyro, data2_acc = process_dataframe(data_n2+'.csv')
    plot_line_acceleration(data1_acc, data_n1)
    plot_line_acceleration(data2_acc, data_n2)
    plot_line_gyro(data1_gyro, data_n1)
    plot_line_gyro(data2_gyro, data_n2)
    plot_spec(data1_gyro, data_n1, True)
    plot_spec(data2_gyro, data_n2, True)
    plot_spec(data1_acc, data_n1+'_acc', False)
    plot_spec(data2_acc, data_n2+'_acc', False)
    
    data_n1 = 'right_41'
    data_n2 = 'right_39'
    data_name1 = 'left_15.csv'
    data_name2 = 'left_07.csv'
    data1_gyro, data1_acc = process_dataframe(data_n1+'.csv')
    data2_gyro, data2_acc = process_dataframe(data_n2+'.csv')
    plot_line_acceleration(data1_acc, data_n1)
    plot_line_acceleration(data2_acc, data_n2)
    plot_line_gyro(data1_gyro, data_n1)
    plot_line_gyro(data2_gyro, data_n2)
    plot_spec(data1_gyro, data_n1, True)
    plot_spec(data2_gyro, data_n2, True)
    plot_spec(data1_acc, data_n1+'_acc', False)
    plot_spec(data2_acc, data_n2+'_acc', False)


    data_n1 = 'up_41'
    data_n2 = 'up_39'
    data_name1 = 'left_15.csv'
    data_name2 = 'left_07.csv'
    data1_gyro, data1_acc = process_dataframe(data_n1+'.csv')
    data2_gyro, data2_acc = process_dataframe(data_n2+'.csv')
    plot_line_acceleration(data1_acc, data_n1)
    plot_line_acceleration(data2_acc, data_n2)
    plot_line_gyro(data1_gyro, data_n1)
    plot_line_gyro(data2_gyro, data_n2)
    plot_spec(data1_gyro, data_n1, True)
    plot_spec(data2_gyro, data_n2, True)
    plot_spec(data1_acc, data_n1+'_acc', False)
    plot_spec(data2_acc, data_n2+'_acc', False)


    data_n1 = 'down_41'
    data_n2 = 'down_39'
    data_name1 = 'left_15.csv'
    data_name2 = 'left_07.csv'
    data1_gyro, data1_acc = process_dataframe(data_n1+'.csv')
    data2_gyro, data2_acc = process_dataframe(data_n2+'.csv')
    plot_line_acceleration(data1_acc, data_n1)
    plot_line_acceleration(data2_acc, data_n2)
    plot_line_gyro(data1_gyro, data_n1)
    plot_line_gyro(data2_gyro, data_n2)
    plot_spec(data1_gyro, data_n1, True)
    plot_spec(data2_gyro, data_n2, True)
    plot_spec(data1_acc, data_n1+'_acc', False)
    plot_spec(data2_acc, data_n2+'_acc', False)


#hw_responses()