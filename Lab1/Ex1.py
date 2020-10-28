'''
Python script which collects the temperature and humidity using a Raspberry Pi module and a DHT-11 sensor.
Wiring:
-Wire the ground and the 5V pinout to the DHT-11 sensor. (It may be useful to use command 'pinout' to check your Raspberry Pi pins)
-Wire the output sensor to the D4 pin
Input:
-The program requires 3 parameters: the duration of the sensoring (time in seconds), the frequency (in s), the name of the output file
Output:
-A file containing a header and a number of lines which depends on the duration and frequency of the measurement. Each line contains the time stamp (dd/mm/yyyy), the time (hh:mm:ss), the temp value and the humi value
'''

import sys
from board import D4
import adafruit_dht
import time
from datetime import datetime
import os

#Save the input parameters in variables
if(len(sys.argv)==3+1):
	duration = float(sys.argv[1])
	freq = float(sys.argv[2])
	file_name = str(sys.argv[3])
else:
	print('WRONG INPUT!! USAGE: time(s), freq(s), output file')
	exit(0)

#Print the current input parameters
print(f'Duration= {duration}, Freq= {freq}, File Name= {file_name}')

#If True it will print the value of temp and humi at each iteration
debug = True

#Create and instance of the DHT11 sensor, using D4 as input pin
dht_device = adafruit_dht.DHT11(D4)

#Open a new file a write the header
file =  open(file_name, 'w')
file.write(f'date,time,temp,humi\n')

#Calculate the number of iteration (how many times we collect data)
cycles = int(duration/freq)
for i in range(cycles):
	#Read the value of temp and humi. Sometimes it raises an exception, it is handled set the value to -1
	try:
		temperature = dht_device.temperature
	except:
		temperature = -1
	try:
		humidity  = dht_device.humidity
	except:
		humidity = -1
	
	#Get the current time 
	now = datetime.now()
	year = now.year
	month = now.month
	day = now.day
	hour = now.hour
	minute = now.minute
	second = now.second
	
	#Save a variable with the correct formatting
	output_str = '{:02d}/{:02d}/{:04d},{:02d}:{:02d}:{:02d},{},{}'.format(day,month,year,hour,minute,second,temperature,humidity)
	if(debug):	
		print(output_str)
	file.write(output_str+'\n')
	time.sleep(freq)
file.close()
