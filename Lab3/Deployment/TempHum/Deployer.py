import argparse
import tensorflow as tf
import os

saved_model_dir = 'model'
converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
tflite_model = converter.convert()

tflite_model_dir = saved_model_dir + '_tflite'
with open(tflite_model_dir, 'wb') as fp:
    fp.write(tflite_model)

size_tflite_model = os.path.getsize(tflite_model_dir)
print(f'Tflite Model size: {size_tflite_model}')

