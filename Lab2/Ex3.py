'''
Python script that calculate the Mel-Frequency Cepstral Coefficients(MFCC) of a given spectrogram.
Input:
-The program requires 2 parameters: an input .spect file, the name of the output file
Output:
-A .mfccs file
'''

import tensorflow as tf
import os
import argparse

class MFCC:
	def __init__(self,mel_bins,sampling_rate,low_freq,up_freq,verbose=False):
		self.num_mel_bins = mel_bins
		self.sampling_rate = sampling_rate
		self.lower_frequency = low_freq
		self.upper_frequency = up_freq
		self.verbose=verbose

	def CalculateMFCC(self,input_file,output_file):
		spectrogram = tf.io.read_file(input_file)
		spectrogram = tf.io.parse_tensor(spectrogram,out_type=tf.float32)

		#print(spectrogram.shape)
		num_spectrogram_bins = spectrogram.shape[-1]
		#print(num_spectrogram_bins)
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

		mfccs = tf.signal.mfccs_from_log_mel_spectrograms(log_mel_spectrogram)[:,:10]

		if(self.verbose):
			print(f'Saved {output_file}')
		file = open(output_file,'w')
		print(mfccs.numpy(),file=file)
		file.close()
		if(self.verbose):
			file_inp_size = os.path.getsize(input_file)
			print(f'File size {file_inp_size}')
			file_out_size = os.path.getsize(output_file)
			print(f'File size {file_out_size}')

if __name__=='__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('-i',type=str, help='input spectrogram file', required=True)
	parser.add_argument('-o',type=str, help='output file', required=True)
	args = parser.parse_args()

	mfcc = MFCC(40,16000,20,4000)
	mfcc.CalculateMFCC(args.i,args.o)
