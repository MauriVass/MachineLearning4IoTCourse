'''
Python script which collects the temperature and humidity using a Raspberry Pi module and a DHT-11 sensor.
Wiring:
-Wire the ground and the 5V pinout of the board to the DHT-11 sensor. (It may be useful to use command 'pinout' to check your Raspberry Pi pins configuration)
-Wire the output sensor to the D4 pin on the board
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
import argparse

class DHT_11:
	def __init__(self,duration,freq,output):
		#Save the input parameters in variables
		self.duration = duration
		self.freq = freq
		self.file_name = output

		#Print the current input parameters
		print(f'Duration= {duration}, Freq= {freq}, File Name= {file_name}')

		#If True it will print the value of temp and humi at each iteration
		self.debug = True

		#Create and instance of the DHT11 sensor, using D4 as input pin
		self.dht_device = adafruit_dht.DHT11(D4)

	def StartSensoring(self):
		#Open a new file a write the header
		file =  open(self.file_name, 'w')
		file.write(f'date,time,temp,humi\n')

		#Calculate the number of iteration (how many times we collect data)
		cycles = int(self.duration/self.freq)
		for i in range(cycles):
			#Read the value of temp and humi. Sometimes it raises an exception, it is handled set the value to -1
			try:
				temperature = self.dht_device.temperature
			except:
				temperature = -1
			try:
				humidity  = self.dht_device.humidity
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
			if(self.debug):
				print(output_str)
			file.write(output_str+'\n')
			time.sleep(self.freq)
		file.close()

if __name__=='__main__':

	#Create a parser obj to get input parameters
	parser = argparse.ArgumentParser()
	parser.add_argument('--duration', type=int, help='duration in s', required=True)
	parser.add_argument('--frequency', type=int, help='sampling rate in Hz', required=True)
	parser.add_argument('-o', type=str, help='output', required=True)
	args = parser.parse_args()

	#Save the input parameters in variables
	duration = args.duration
	freq = args.frequency
	file_name = args.o

	#Create sensor instance
	dht_sensor = DHT_11(duration,freq,file_name)
	#Start sensoring temp and humidity
	dht_sensor.StartSensoring()
