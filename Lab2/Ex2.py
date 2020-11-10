
'''
Python script that calculates the Short Time Fourier Transform (STFT) of an audio signal and save it as an image.
Input:
-The program requires 2 parameters:
		-an input .wav file,
		-the window frame length (in seconds),
		-the window stride length (in seconds),
		-the name of the output file.
Output:
-A .png file
'''

import tensorflow as tf
import numpy as np
import time
import os
import argparse

class STFT:
	def CalculateSTFT(self,input_file,frame_length,stride,output_file):
		#Array of bytes
		audio = tf.io.read_file(input_file)

		#Signal and frequency of the audio input
		tf_audio, rate = tf.audio.decode_wav(audio)
		#Add a dimension to specify the number of channels
		tf_audio = tf.squeeze(tf_audio, 1)

		#Hyperparameters. (Maybe) Good values: 0.04, 0.02
		frame_length = float(frame_length)
		stride = float(stride)
		frame_length = int(rate.numpy() * frame_length)
		frame_step = int(rate.numpy() * stride)
		print(f'Frame length: {frame_length}, frame step: {frame_step}')

		start_time = time.time()
		#Calculate the STFT of the signal given frame_length and frame_step
		stft = tf.signal.stft(tf_audio,
					frame_length=frame_length,
					frame_step=frame_step,
					fft_length=frame_length)
		#Transform the complex number in real number
		spectrogram = tf.abs(stft)
		end_time = time.time()
		elapsed_time = end_time - start_time
		print(f'Required time: {elapsed_time:.3f}')

		byte_string = tf.io.serialize_tensor(spectrogram)
		#Save the spectrogram and the .png image on disk
		output_file = f'{output_file}.spect'
		tf.io.write_file(output_file,byte_string)

		#The file size of the output(spectrogram) should be more than the input file(wav)
		print(f'Input file size {os.path.getsize(input_file)}')
		print(f'Output file size {os.path.getsize(output_file)}')

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
		#Save image as .png file
		tf.io.write_file(f'{output_file}.png',png_image)

if __name__=='__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('-i', type=str, help='input .wav file', required=True)
	parser.add_argument('--frame', type=str, help='the window frame size (in seconds)', required=True)
	parser.add_argument('--stride', type=str, help='the window stride size (in seconds)', required=True)
	parser.add_argument('-o', type=str, help='output file (no extension)', required=True)
	args = parser.parse_args()

	input = args.i
	frame_size = args.frame
	stride = args.stride
	output = args.o
	stft = STFT()
	stft.CalculateSTFT(input,frame_size,stride,output)

