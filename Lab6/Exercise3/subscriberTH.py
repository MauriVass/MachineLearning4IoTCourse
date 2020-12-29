import time
import sys
sys.path.insert(0, './../Exercise1')
from MyMQTT import MyMQTT
import json

class DoSomething():
	def __init__(self, clientID):
		# create an instance of MyMQTT class
		self.clientID = clientID
		self.myMqttClient = MyMQTT(self.clientID, "mqtt.eclipseprojects.io", 1883, self) 
		
		self.counter = 0

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
		r = msg.decode('utf-8')
		r = json.loads(r)
		events = r['events']
		audio = events['audio']

		file_name = 'audio'+str(self.counter)+'.wav'
		waveFile = wave.open(file_name,'wb')
    	waveFile.setnchannels(1)
		waveFile.setsampwidth(audio.get_sample_size(pyaudio.paInt16))
		waveFile.setframerate(48000)
		#Merge all the recorded frames as binary
		waveFile.writeframes(b''.join(audio['vd']))
		waveFile.close()
		print('Stored: ',file_name)

if __name__ == "__main__":
	test = DoSomething("subscriber 1")
	test.run()
	test.myMqttClient.mySubscribe("/ASD123/date/time/timestamp/Sensor1/")

	a = 0
	while (a < 30):
		if(a%5):
			message = 'Record Audio, please <3'
			test.myMqttClient.myPublish ("/ASD123/date/time/timestamp/Sensor1/", json.dumps(message), False)
		a += 1
		time.sleep(4)

	test.end()