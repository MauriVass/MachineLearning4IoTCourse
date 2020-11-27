import tensorflow as tf
from DHT_11 import DHT_11

sensor = DHT_11(7,1)

input_data = sensor.StartSensoring()
print(input_data)
input_data = tf.constant(input_data,dtype=tf.float32)
print(input_data)
#input_data = input_data.flatten()

interpreter = tf.lite.Interpreter(model_path='model_tflite')
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
input_details = interpreter.get_output_details()

interpreter.set_tensor(input_details[0]['index'], input_data)
interpreter.invoke()
my_output = interpreter.get_tensor(output_details[0]['index'])
