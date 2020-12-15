import tensorflow as tf
from DHT_11 import DHT_11
import numpy as np
import zlib
import os

print('----	---	---	----')
print('REQUIREMENTS:')
print('a) T MAE < 0.5°C and Rh MAE < 1.8% and TFLite Size < 2 kB')
print('b) T MAE < 0.6 °C and Rh MAE < 1.9% and TFLite Size < 1.7 kB')
print('----	---	---	----\n')


#Decompress
model_path_a = 'Models/TH_MLPalpha0_25spars0_9.tflite_W.zip'
model_path_b = 'Models/THmodelMLP.tflite.zip'
model_path = model_path_a
print(f'Model Size: {(os.path.getsize(model_path)/1024):.2f} Kb')
with open(model_path, 'rb') as fp:
	model = zlib.decompress(fp.read())
	output_model = model_path[:-4]
	file = open(output_model,'wb')
	print('Saving: ',output_model)
	file.write(model)
	file.close()

interpreter = tf.lite.Interpreter(model_path=output_model)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

eval_from_sensor = False
mae = [0,0]
if(eval_from_sensor):
	input = 6
	output = 6
	sensor = DHT_11(input+output,1)

	samples = 1
	for i in range(samples):
		input_data = sensor.StartSensoring()

		x = input_data[:input]
		if(output>1):
			y = input_data[input:input+output]
		else:
			y = input_data[-1]

		input_data = tf.constant(x,dtype=tf.float32)
		input_data = tf.expand_dims(input_data, 0)

		interpreter.set_tensor(input_details[0]['index'], input_data)
		interpreter.invoke()
		my_output = interpreter.get_tensor(output_details[0]['index'])[0]

		error = tf.abs(my_output-y)
		mae = tf.reduce_mean(error, axis=(0,))
		print(y)
		print(my_output)
		print(error)
		print(mae)
		#print(f'Measured: {y[0]}, {y[1]} --- Predicted: {my_output[0]:.2f}, {my_output[1]:.2f} --- MAE: {mae_t:.2f}, {mae_h:.2f}')
else:
	tensor_specs = (tf.TensorSpec([None, 6, 2], dtype=tf.float32),tf.TensorSpec([None, 6, 2]))
	test_ds = tf.data.experimental.load('./th_test1', tensor_specs)
	test_ds = test_ds.unbatch().batch(1)

	samples = 0

	for x,y in test_ds:
		input_data = x
		y_true = y.numpy()[0]

		interpreter.set_tensor(input_details[0]['index'], input_data)
		interpreter.invoke()
		my_output = interpreter.get_tensor(output_details[0]['index'])[0]

		samples+=1
		error = tf.abs(my_output-y_true)
		mae += tf.reduce_mean(error, axis=(0,))
#print(f'\nMAE T: {(mae[0]/samples):.4f}, H: {(mae[1]/samples):.4f}')
print(f'MAE: temp: {(mae[0]/samples):.4f}, humi: {(mae[1]/samples):.4f}')
