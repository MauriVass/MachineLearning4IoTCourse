'''
Python script that calculate the Mel-Frequency Cepstral Coefficients(MFCC) of a given spectrogram.
Input:
-The program requires 2 parameters:
	-an input .spect file,
	-number of mel filters
	-number of coefficients
	-the name of the output file
Output:
-A .mfccs file
'''

import tensorflow as tf
import os
import argparse
import time

class MFCC:
	def __init__(self,mel_bins,coefficients,sampling_rate,low_freq,up_freq):
		self.num_mel_bins = mel_bins
		self.coefficients = coefficients
		self.sampling_rate = sampling_rate
		self.lower_frequency = low_freq
		self.upper_frequency = up_freq

	def CalculateMFCC(self,input_file,output_file):
		#Read file
		spectrogram = tf.io.read_file(input_file)
		spectrogram = tf.io.parse_tensor(spectrogram,out_type=tf.float32)
		print(f'Spectrogram shape: {spectrogram.shape}')

		num_spectrogram_bins = spectrogram.shape[-1]
		start_time = time.time()
		linear_to_mel_weight_matrix = tf.signal.linear_to_mel_weight_matrix(
					self.num_mel_bins,
					num_spectrogram_bins,
					self.sampling_rate,
					self.lower_frequency,
					self.upper_frequency)
		mel_spectrogram = tf.tensordot(spectrogram,linear_to_mel_weight_matrix,1)
		mel_spectrogram.set_shape(spectrogram.shape[:-1].concatenate(
					linear_to_mel_weight_matrix.shape[-1:]))
		log_mel_spectrogram = tf.math.log(mel_spectrogram + 1e-6)

		#Not all coefficients are important, so it is useful to select only some of them.
		#To find them you can use a search algorithm
		mfccs = tf.signal.mfccs_from_log_mel_spectrograms(log_mel_spectrogram)[:,:self.coefficients]
		print('MFCCS shape: ', mfccs.shape)
		end_time = time.time()
		print(f'Execution time: {(end_time-start_time):.3f}')

		print(f'Saved {output_file}.mfccs')
		file = open(output_file+'.mfccs','w')
		print(mfccs.numpy(),file=file)
		file.close()
		file_inp_size = os.path.getsize(input_file)
		print(f'File size {file_inp_size}')
		file_out_size = os.path.getsize(output_file+'.mfccs')
		print(f'File size {file_out_size}')

		image = tf.transpose(mfccs)
		#Add the 'channel' dimension
		image = tf.expand_dims(image,-1)
		#Normalize to have values on a range [0,255]
		min_val = tf.reduce_min(image)
		max_val = tf.reduce_max(image)
		image = (image-min_val) / (max_val-min_val)
		image = image * 255
		image = tf.cast(image,tf.uint8)

		png_image = tf.io.encode_png(image)
		tf.io.write_file(f'{output_file}.png',png_image)

if __name__=='__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('-i',type=str, help='input spectrogram file', required=True)
	parser.add_argument('--filters',type=int, help='input spectrogram file', required=True)
	parser.add_argument('--coefficients',type=int, help='input spectrogram file', required=True)
	parser.add_argument('-o',type=str, help='output file', required=True)
	args = parser.parse_args()

	mel_filters = args.filters #40
	coefficients = args.coefficients #10
	sampling_resolution = 16000 #from previous pre-processing steps
	low_freq = 20
	high_freq = 4000
	mfcc = MFCC(mel_filters,coefficients,16000,20,4000)
	mfcc.CalculateMFCC(args.i,args.o)
