import time
import json
import sys
sys.path.insert(0, './../Exercise1')
from DoSomething import DoSomething

class Receiver(DoSomething):
	def notify(self, topic, msg):
		# manage here your received message. You can perform some error-check here
		r = msg.decode('utf-8')
		r = json.loads(r)
		events = r['e']
		if(topic.find('temperature')>=0):
			temperature = events[0]
			text = f"Temperature: {temperature['v']}{temperature['u']}"
		else:
		# if(len(events)>1):
			humidity = events[0]
			text = f"Humidity: {humidity['v']}{humidity['u']}"
		print(text)

if __name__ == "__main__":
	test = Receiver("Subscriber TH")
	test.run()
	test.myMqttClient.mySubscribe("/ABCDE12345/Sensor1/temperature/")
	test.myMqttClient.mySubscribe("/ABCDE12345/Sensor1/humidity/")

	a = 0
	while (a < 20):
		a += 1
		time.sleep(1)

	test.end()