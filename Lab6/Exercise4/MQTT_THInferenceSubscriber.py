import time
import sys
import json
import tensorflow as tf
sys.path.insert(0, './../Exercise1')
from DoSomething import DoSomething

class Receiver(DoSomething):
	def notify(self, topic, msg):
		# manage here your received message. You can perform some error-check here
		# print(topic,msg)
		r = msg.decode('utf-8')
		r = json.loads(r)
		events = r['e']

		if(self.temp_avaible==False or self.hum_avaible==False):
			for e in events:
				if(e['n'] == 'temperature'):
					self.temp_samples.append(e['v'])
					self.temp_avaible = True
				elif(e['n'] == 'humidity'):
					self.hum_samples.append(e['v'])
					self.hum_avaible = True

		if(self.temp_avaible and self.hum_avaible):
			data = []
			for t,h in zip(self.temp_samples,self.hum_samples):
				data.append([t,h])
			prediction = self.Predict(model_type='',data=data)
			print(f'Prediction. Temp: {prediction[0]}, Hum: {prediction[1]}')
			

	def Predict(self,model_type,data):
		saved_model_dir='THFmodelCNN_tflite'

		interpreter = tf.lite.Interpreter(model_path=saved_model_dir)
		interpreter.allocate_tensors()

		input_details = interpreter.get_input_details()
		output_details = interpreter.get_output_details()

		mean = tf.constant( [[[ 9.107597, 75.904076]]], shape=(1, 1, 2), dtype=tf.float32 )
		std = tf.constant( [[[ 8.654227, 16.557089]]], shape=(1, 1, 2), dtype=tf.float32 )
		features = (data - mean) / (std + 1.e-6)
	
		interpreter.set_tensor(input_details[0]['index'], features)
		interpreter.invoke()
		predict = interpreter.get_tensor(output_details[0]['index'])[0]

		self.temp_samples = []
		self.temp_avaible = False
		self.hum_samples = []
		self.hum_avaible = False
		return predict

if __name__ == "__main__":
	test = Receiver("Subscriber TH")
	test.run()
	test.myMqttClient.mySubscribe("/ABCDE12345/Sensor1/temperature/")
	test.myMqttClient.mySubscribe("/ABCDE12345/Sensor1/humidity/")

	a = 0
	while (a < 200):
		a += 1
		time.sleep(1)

	test.end()