import time
import datetime
import json
from board import D4
import adafruit_dht
import pyaudio
import wave
import base64

sys.path.insert(0, './../Exercise1')
from DoSomething import DoSomething

class Receiver(DoSomething):
	def notify(self, topic, msg):
		# manage here your received message. You can perform some error-check here
		print(topic,msg)
		if(topic==self.myMqttClient._topic):
			r = msg.decode('utf-8')
			r = json.loads(r)
			print(f"Record Audio ", topic, r)
			self.record_audio = True

class DataCollector():
	def __init__(self):
		self.dht_device = adafruit_dht.DHT11(D4)

		self.is_mic_plugged = True
		if(self.is_mic_plugged):
			self.audio = pyaudio.PyAudio()
			self.stream = self.audio.open(format=pyaudio.paInt16,
				channels=1,
				rate= 48000, input=True,
				frames_per_buffer=4800)
			self.stream.stop_stream()


	def GetTemp(self):
		#Collect Temp and Hum
		temperature = self.dht_device.temperature
		return temperature

	def GetHumi(self):
		humidity = self.dht_device.humidity
		return humidity

	def GetAudio(self):
		#Record file audio
		frames = []
		self.stream.start_stream()
		for i in range(10):
			data = self.stream.read(4800) #, exception_on_overflow=False)
			frames.append(data)
		self.stream.stop_stream()
		#It is needed to send data over network since you can't send raw bytes
		audio_b64bytes = base64.b64encode(b''.join(frames))
		audio_string = audio_b64bytes.decode()
		return audio_string


if __name__ == "__main__":
	test = Receiver("Publisher THA")
	test.run()
	idtopic = '/ABCDE12345/'
	test.myMqttClient.mySubscribe(idtopic+'Sensor1/record/')
	dc = DataCollector()

	a = 0

	#This is used to publish or not the timestamp (==2 -> publish)
	counter = 2
	ip = 'http://169.254.37.210/'
	while (a < 10):
		date_time = datetime.datetime.now()
		timestamp = datetime.datetime.timestamp(date_time)

		date_str = str(date_time.date())
		time_str = str(date_time.time()).split('.')[0]

		events = []
		temperature = dc.GetTemp()
		events.append({'n':'temperature', 'u':'Cel', 't':0, 'v':temperature})

		if(counter==2):
			humidity = dc.GetHumi()
			events.append({'n':'humidity', 'u':'%RH', 't':0, 'v':humidity})
			counter=0

		if(test.record_audio):
			audio_string= None
			if(dc.is_mic_plugged):
				audio_string = dc.GetAudio()
			events.append({'n':'audio', 'u':'/', 't':0, 'vd':audio_string})
			test.record_audio = False

		for e in events:
			body = {
						'bn' : ip,
						'bi' : int(timestamp),
						'e' : [e]
					}
			test.myMqttClient.myPublish (idtopic+"Sensor1/"+e['n']+'/', json.dumps(body),False)

		counter += 1
		a+=1
		time.sleep(10)

	test.end()
