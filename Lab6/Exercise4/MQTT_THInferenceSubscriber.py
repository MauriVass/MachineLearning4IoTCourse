import time
import sys
sys.path.insert(0, './../Exercise1')
from MyMQTT import MyMQTT
import json
#import tensorflow as tf

class DoSomething():
	def __init__(self, clientID):
		# create an instance of MyMQTT class
		self.clientID = clientID
		self.myMqttClient = MyMQTT(self.clientID, "mqtt.eclipseprojects.io", 1883, self) 
		
		self.tep_samples = []
		self.hum_samples = []

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
			self.tep_samples = events
		else:
			self.hum_samples = events
		print(self.tep_samples,self.hum_samples)

		data = []
		for t,h in zip(self.tep_samples,self.hum_samples):
			data.appned(t,h)
		prediction = Prediction(data=data)
		print(prediction)

def Predict(self,model_type='',data):
	model = tf.keras.models.load_model('Models/THFmodelCNN')
	# Show the model architecture
	new_model.summary()

	#tf.Tensor([[[ 9.107597 75.904076]]], shape=(1, 1, 2), dtype=float32) tf.Tensor([[[ 8.654227 16.557089]]], shape=(1, 1, 2), dtype=float32)
	mean = tf.Tensor([[[ 9.107597 75.904076]]], shape=(1, 1, 2), dtype=float32)
	std = tf.Tensor([[[ 8.654227 16.557089]]], shape=(1, 1, 2), dtype=float32)
	features = (data - mean) / (std + 1.e-6)
	prediction = new_model.predict(test_images)
	return predict


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