import time
import datetime
import json
from board import D4
import adafruit_dht
import pyaudio
import wave
import base64

import sys
sys.path.insert(0, './../Exercise1')
from MyMQTT import MyMQTT


class DoSomething():
	def __init__(self, clientID):
		# create an instance of MyMQTT class
		self.clientID = clientID
		self.myMqttClient = MyMQTT(self.clientID, "mqtt.eclipseprojects.io", 1883, self)

		self.record_audio = False

	def run(self):
		# if needed, perform some other actions befor starting the mqtt communication
		print ("running %s" % (self.clientID))
		self.myMqttClient.start()

	def end(self):
		# if needed, perform some other actions befor ending the software
		print ("ending %s" % (self.clientID))
		self.myMqttClient.stop ()

	def notify(self, topic, msg):
		# manage here your received message. You can perform some error-check here
		if(topic==self.myMqttClient._topic):
			r = msg.decode('utf-8')
			r = json.loads(r)
			print(f"Record Audio ", topic, r)
			self.record_audio = True

class DataCollector():
	def __init__(self):
		self.dht_device = adafruit_dht.DHT11(D4)

		self.is_mic_plugged = False
		if(self.is_mic_plugged):
			self.audio = pyaudio.PyAudio()
			self.stream = self.audio.open(format=pyaudio.paInt16,
				channels=1,
				rate= 48000, input=True,
				frames_per_buffer=4800)
			self.stream.wait()


	def GetTemp(self):
		#Collect Temp and Hum
		temperature = self.dht_device.temperature
		return temperature

	def GetHumi(self):
		humidity = self.dht_device.humidity
		return humidity

	def GetAudio(self):
		if(self.is_mic_plugged):
			#Record file audio
			frames = []
			self.stream.start_stream()
			for i in range(10):
				data = stream.read(4800) #, exception_on_overflow=False)
				frames.append(data)
			self.stream.stop_stream()
			#It is needed to send data over network since you can't send raw bytes
			audio_64_bytes = base64.b64encode(b''.join(frames))
			audio_string = audio_b64bytes.decode()
			return audio_string


if __name__ == "__main__":
	test = DoSomething("publisher 1")
	test.run()
	test.myMqttClient.mySubscribe("/ASD123/date/time/timestampa/Sensor1/Audio/")
	dc = DataCollector()

	a = 0

	#This is used to publish or not the timestamp (==2 -> publish)
	counter = 2
	while (a < 4):
		date_time = datetime.datetime.now()
		timestamp = datetime.datetime.timestamp(date_time)

		date_str = str(date_time.date())
		time_str = str(date_time.time()).split('.')[0]
		message = {'date':date_str}
		test.myMqttClient.myPublish ("/ASD123/date/", json.dumps(message), False)
		message['time'] = time_str
		test.myMqttClient.myPublish ("/ASD123/date/time/", json.dumps(message),False)
		message['timestamp'] = timestamp
		test.myMqttClient.myPublish ("/ASD123/date/time/timestamp/", json.dumps(message),False)


		events = []
		temperature = dc.GetTemp()
		events.append({'n':'temperature', 'u':'Cel', 't':0, 'v':temperature})

		if(counter==2):
			humidity = dc.GetHumi()
			events.append({'n':'humidity', 'u':'%RH', 't':0, 'v':humidity})
			counter=0

		if(test.record_audio):
			audio_string = dc.GetAudio()
			events.append({'n':'audio', 'u':'/', 't':0, 'vb':audio_string})
			test.record_audio = False

		ip = '2.44.137.33' + '/'
		body = {
					'bn' : 'http://'+ip,
					'bi' : int(timestamp),
					'e' : events
				}

		test.myMqttClient.myPublish ("/ASD123/date/time/timestamp/Sensor1/", json.dumps(body),False)
		counter += 1
		time.sleep(10)

	test.end()
