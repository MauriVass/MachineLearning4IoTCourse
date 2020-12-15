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
	def __init__(self,duration,freq):
		#Save the input parameters in variables
		self.duration = duration
		self.freq = freq

		#Print the current input parameters
		print(f'Duration= {duration}, Freq= {freq}')

		#If True it will print the value of temp and humi at each iteration
		self.debug = True

		#Create and instance of the DHT11 sensor, using D4 as input pin
		self.dht_device = adafruit_dht.DHT11(D4)

	def StartSensoring(self):
		#Calculate the number of iteration (how many times we collect data)
		cycles = int(self.duration/self.freq)
		values = []
		for i in range(cycles):
			#Read the value of temp and humi. Sometimes it raises an exception, it is handled set the value to -1
			try:
				#Remember to apply the safe preprocessing step of the training. Both temp and hum
				temperature = (self.dht_device.temperature - 9.11) / (8.65 + 1.e-6) if i<6 else self.dht_device.temperature
			except:
				#To avoid large MAE, substitute -1 with the mean
				temperature = 9.11
			try:
				humidity  = (self.dht_device.humidity - 75.9) / (16.55 + 1.e-6) if i<6 else self.dht_device.humidity
			except:
				humidity = 75.9
			values.append([temperature,humidity])

			time.sleep(self.freq)
		return values
