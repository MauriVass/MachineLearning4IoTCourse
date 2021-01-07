import time
import sys
import json
import wave
import pyaudio
import datetime

sys.path.insert(0, './../Exercise1')
from DoSomething import DoSomething

class Receiver(DoSomething):
	def notify(self, topic, msg):
		# manage here your received message. You can perform some error-check here 
		# print(topic, msg)
		if(topic.find('audio')>=0): 
			r = msg.decode('utf-8')
			r = json.loads(r)
			events = r['e']
			audio = events[0]

			file_name = 'audio'+str(self.counter)+'.wav'
			waveFile = wave.open(file_name,'wb')
			waveFile.setnchannels(1)
			waveFile.setsampwidth(pyaudio.paInt16)
			waveFile.setframerate(48000)
			#Merge all the recorded frames as binary
			waveFile.writeframes(b''.join(audio['vd']))
			waveFile.close()
			print('Stored: ',file_name)

if __name__ == "__main__":
	test = Receiver("Subscriber Audio")
	test.run()
	test.myMqttClient.mySubscribe("/ABCDE12345/Sensor1/audio/")

	a = 0
	ip = 'http://169.254.37.210/'
	while (a < 20):
		if(a%3):
			timestamp = datetime.datetime.timestamp(date_time)
			body = {
						'bn' : ip,
						'bi' : int(timestamp),
						'e' : [{'n':'audio', 'u':'/', 't':0, 'vd': None}]
					}
			test.myMqttClient.myPublish("/ABCDE12345/Sensor1/record", json.dumps(body),False)
		a += 1
		time.sleep(1)

	test.end()