import time
import datetime
import json
from board import D4
import adafruit_dht

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
	test = DoSomething("Publisher TH")
	test.run()
	idtopic = '/ABCDE12345/'
	dc = DataCollector()

	a = 0

	while (a < 10):
		date_time = datetime.datetime.now()
		timestamp = datetime.datetime.timestamp(date_time)

		date_str = str(date_time.date())
		time_str = str(date_time.time()).split('.')[0]
		message = {'date':date_str}
		#test.myMqttClient.myPublish (idtopic+"date/", json.dumps(message), False)
		message['time'] = time_str
		#test.myMqttClient.myPublish (idtopic+"date/time/", json.dumps(message),False)
		message['timestamp'] = int(timestamp)
		#test.myMqttClient.myPublish (idtopic+"date/time/timestamp/", json.dumps(message),False)


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

		ip = '2.44.137.33' + '/'
		body = {
					'bn' : 'http://'+ip,
					'bi' : int(timestamp),
					'e' : tem_events
				}
		test.myMqttClient.myPublish (idtopic+"date/time/timestamp/Sensor1/temperature/", json.dumps(body),False)
		body = {
					'bn' : 'http://'+ip,
					'bi' : int(timestamp),
					'e' : hum_events
				}
		test.myMqttClient.myPublish (idtopic+"date/time/timestamp/Sensor1/humidity/", json.dumps(body),False)

		a+=1

	test.end()
