import tensorflow as tf
import numpy as np
import time
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-i', type=str, help='input file', required=True)
args = parser.parse_args()

input_file = args.i
audio = tf.io.read_file(input_file)

tf_audio, rate = tf.audio.decode_wav(audio)
tf_audio = tf.squeeze(tf_audio, 1)

frame_step = 960 #int(rate.numpy() / (0.10 * 1000))
frame_length = int(rate.numpy() * 0.040)
print(f'Frame step: {frame_step}, frame length: {frame_length}')

stft = tf.signal.stft(tf_audio,
			frame_length=frame_length,
			frame_step=frame_step,
			fft_length=frame_length)
start_time = time.time()
spectrogram = tf.abs(stft)
end_time = time.time()

elapsed_time = end_time - start_time
print(f'Required time: {elapsed_time:.3f}')

byte_string = tf.io.serialize_tensor(spectrogram)
output_file = 'output_Ex2'
tf.io.write_file(output_file,byte_string)

print(f'File size {os.path.getsize(output_file)}')

image = tf.transpose(spectrogram)
#Add the 'channel' dimension
image = tf.expand_dims(image,-1)
#Take the logarithm for better visualization
image = tf.math.log(image + 1.e-6)

#Normalize to have values on a range [0,255]
min_val = tf.reduce_min(image)
max_val = tf.reduce_max(image)
image = (image-min_val) / (max_val-min_val)
image = image * 255
image = tf.cast(image,tf.uint8)

png_image = tf.io.encode_png(image)
tf.io.write_file(f'{input_file}.png',png_image)
