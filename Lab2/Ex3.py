import tensorflow as tf
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-i',type=str, help='input spectrogram file', required=True)
args = parser.parse_args()

input_file = args.i
spectrogram = tf.io.read_file(input_file)
spectrogram = tf.io.parse_tensor(spectrogram,out_type=tf.float32)

num_mel_bins = 40
sampling_rate = 16000
lower_frequency = 20
upper_frequency = 4000
print(spectrogram.shape)
num_spectrogram_bins = spectrogram.shape[-1]
print(num_spectrogram_bins)
linear_to_mel_weight_matrix = tf.signal.linear_to_mel_weight_matrix(
			num_mel_bins,
			num_spectrogram_bins,
			sampling_rate,
			lower_frequency,
			upper_frequency)
mel_spectrogram = tf.tensordot(spectrogram,linear_to_mel_weight_matrix,1)
mel_spectrogram.set_shape(spectrogram.shape[:-1].concatenate(
			linear_to_mel_weight_matrix.shape[-1:]))
log_mel_spectrogram = tf.math.log(mel_spectrogram + 1e-6)

mfccs = tf.signal.mfccs_from_log_mel_spectrograms(log_mel_spectrogram)[:,:10]

output_file = f'output_{input_file[:-6]}_Ex3.mfccs'
print(f'Saved {output_file}')
file = open(output_file,'w')
print(mfccs.numpy(),file=file)# tf.write_file
file.close()
file_inp_size = os.path.getsize(input_file)
print(f'File size {file_inp_size}')
file_out_size = os.path.getsize(output_file)
print(f'File size {file_out_size}')
