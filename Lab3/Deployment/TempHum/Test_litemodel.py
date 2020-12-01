import tensorflow as tf
from DHT_11 import DHT_11
import numpy as np


sensor = DHT_11(7,1)

samples = 100
mae = [0,0]
for i in range(samples):
	input_data = sensor.StartSensoring()

	x = input_data[:-1]
	y = input_data[-1]

	input_data = tf.constant(x,dtype=tf.float32)
	input_data = tf.expand_dims(input_data, 0)

	interpreter = tf.lite.Interpreter(model_path='THFmodelMLP_tflite')
	interpreter.allocate_tensors()

	input_details = interpreter.get_input_details()
	output_details = interpreter.get_output_details()
	#print(input_details)
	#print(output_details)

	interpreter.set_tensor(input_details[0]['index'], input_data)
	interpreter.invoke()
	my_output = interpreter.get_tensor(output_details[0]['index'])[0]

	mae_t = np.abs(y[0] - my_output[0])
	mae_h = np.abs(y[1] - my_output[1])
	mae[0] += mae_t
	mae[1] += mae_h
	print(f'Measured: {y[0]}, {y[1]} --- Predicted: {my_output[0]:.2f}, {my_output[1]:.2f} --- MAE: {mae_t:.2f}, {mae_h:.2f}')

print(f'\nMAE T: {(mae[0]/samples):.4f}, H: {(mae[1]/samples):.4f}')
