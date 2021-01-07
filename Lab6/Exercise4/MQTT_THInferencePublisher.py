import time
import datetime
import json
from board import D4
import adafruit_dht

import sys
sys.path.insert(0, './../Exercise1')
from DoSomething import DoSomething

class Receiver(DoSomething):
	def notify(self, topic, msg):
		# manage here your received message. You can perform some error-check here
		print(topic,msg)

class DataCollector():
	def __init__(self):
		self.dht_device = adafruit_dht.DHT11(D4)

	def GetTemp(self):
		#Collect Temp and Hum
		temperature = self.dht_device.temperature
		return temperature

	def GetHumi(self):
		humidity = self.dht_device.humidity
		return humidity

if __name__ == "__main__":
	test = Receiver("Publisher TH")
	test.run()
	idtopic = '/ABCDE12345/'
	dc = DataCollector()

	a = 0

	ip = '2.44.137.33/'
	while (a < 10):
		date_time = datetime.datetime.now()
		timestamp = datetime.datetime.timestamp(date_time)

		date_str = str(date_time.date())
		time_str = str(date_time.time()).split('.')[0]

		tem_events = []
		hum_events = []
		n_sampes = 6
		total_time_sec = 60
		for i in range(n_sampes):
			#Read the value of temp and humi. Sometimes it raises an exception, it is handled set the value to -1
			try:
				temperature = dc.GetTemp()
			except:
				temperature = -1
			try:
				humidity  = dc.GetHumi()
			except:
				humidity = -1
			tem_events.append({'n':'temperature', 'u':'Cel', 't':i, 'v':temperature})
			hum_events.append({'n':'humidity', 'u':'%RH', 't':i, 'v':humidity})
			time.sleep(total_time_sec/n_sampes)

		body = {
				'bn' : 'http://'+ip,
				'bi' : int(timestamp),
				'e' : tem_events
			}
		test.myMqttClient.myPublish (idtopic+"Sensor1/temperature/", json.dumps(body),False)
		body = {
				'bn' : 'http://'+ip,
				'bi' : int(timestamp),
				'e' : hum_events
			}
		test.myMqttClient.myPublish (idtopic+"Sensor1/humidity/", json.dumps(body),False)
		print('Published')

		a+=1

	test.end()

