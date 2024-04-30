import serial
import time
import csv

port_name = '/dev/cu.usbserial-130'
file_to_be_saved = 'right_41.csv'
s = serial.Serial(port_name, 115200)
s.flushInput()
with open(file_to_be_saved, 'a') as file:
    w = csv.writer(file, delimiter=',')
    w.writerow(['time', 'measure'])
    while True:
        line = s.readline()
        line= str(line)
        w.writerow([time.time(), line])
