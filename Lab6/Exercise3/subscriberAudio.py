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
		r = msg.decode('utf-8')
		r = json.loads(r)
		events = r['events']
		temperature = events['temperature']
		text = f'Temperature: {temperature['v']}{temperature['u']}'
		if('humidity' is in events):
			humidity = events['humidity']
			text += f',Humidity: {humidity['v']}{humidity['u']}'
		print(text)

if __name__ == "__main__":
	test = DoSomething("subscriber 2")
	test.run()
	test.myMqttClient.mySubscribe("/ASD123/date/time/timestamp/")

	a = 0
	while (a < 30):
		a += 1
		time.sleep(1)

	test.end()