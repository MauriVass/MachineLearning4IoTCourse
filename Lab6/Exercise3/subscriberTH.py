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
		# print(topic,msg)
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
	test = DoSomething("Subscriber TH")
	test.run()
	test.myMqttClient.mySubscribe("/ABCDE12345/date/time/timestamp/Sensor1/temperature/")
	test.myMqttClient.mySubscribe("/ABCDE12345/date/time/timestamp/Sensor1/humidity/")

	a = 0
	while (a < 20):
		a += 1
		time.sleep(1)

	test.end()