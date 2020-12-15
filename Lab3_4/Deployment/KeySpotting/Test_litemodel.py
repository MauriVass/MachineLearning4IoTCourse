import tensorflow as tf
from Microphone import Mic
import numpy as np

class SignalGenerator:
	def __init__(self, labels, sampling_rate, frame_length, frame_step, num_mel_bins=None, lower_frequency=None, upper_frequency=None, num_coefficients=None, mfcc=False):
		self.labels=labels
		self.sampling_rate=sampling_rate
		self.frame_length=frame_length
		self.frame_step=frame_step
		self.num_mel_bins = num_mel_bins
		self.lower_frequency = lower_frequency
		self.upper_frequency = upper_frequency
		self.num_coefficients = num_coefficients
		self.mfccs=mfcc

		if(mfcc):
			num_spectrogram_bins = frame_length // 2 + 1
			self.linear_to_mel_weight_matrix = tf.signal.linear_to_mel_weight_matrix(
					self.num_mel_bins,
					num_spectrogram_bins,
					self.sampling_rate,
					self.lower_frequency,
					self.upper_frequency)
			self.preprocess = self.preprocess_with_mfcc
		else:
			self.preprocess = self.preprocess_with_stft


	def read(self, file_path):
		parts = tf.strings.split(file_path, os.path.sep)
		label = parts[-2]
		label_id = tf.argmax(label == self.labels)
		audio_bynary = tf.io.read_file(file_path)
		audio, _ = tf.audio.decode_wav(audio_bynary)
		#print('Sampling: ', np.array(r))
		audio = tf.squeeze(audio, axis=1)
		return audio, label_id

	def pad(self, audio):
		zero_padding = tf.zeros(self.sampling_rate - tf.shape(audio), dtype=tf.float32)
		audio = tf.concat([audio,zero_padding],0)
		audio.set_shape([self.sampling_rate])
		return audio

	def get_spectrogram(self, audio):
		#Calculate the STFT of the signal given frame_length and frame_step
		stft = tf.signal.stft(audio,
						frame_length=self.frame_length,
						frame_step=self.frame_step,
						fft_length=self.frame_length)
		#Transform the complex number in real number
		spectrogram = tf.abs(stft)
		return spectrogram

	def get_mfccs(self, spectrogram):
		mel_spectrogram = tf.tensordot(spectrogram,
						self.linear_to_mel_weight_matrix, 1)
		log_mel_spectrogram = tf.math.log(mel_spectrogram + 1.e-6)
		mfccs = tf.signal.mfccs_from_log_mel_spectrograms(log_mel_spectrogram)
		mfccs = mfccs[:, :self.num_coefficients]
		return mfccs

	def preprocess_with_stft(self, file_path):
		audio, label = self.read(file_path)
		audio = self.pad(audio)
		spectrogram = self.get_spectrogram(audio)
		spectrogram = tf.expand_dims(spectrogram, -1)
		spectrogram  = tf.image.resize(spectrogram, [32,32])
		return spectrogram, label

	def preprocess_with_mfcc(self, file_path):
		audio, label = self.read(file_path)
		audio = self.pad(audio)
		spectrogram = self.get_spectrogram(audio)
		mfccs = self.get_mfccs(spectrogram)
		mfccs = tf.expand_dims(mfccs, -1)
		return mfccs, label

	def make_dataset(self, file, train=False):
		#This method creates a dataset from a numpy array (our listfile path)
		ds = file
		#Different preprocess step depending on the input parameter
		ds = ds.map(self.preprocess, num_parallel_calls=4)
		return ds

sensor = Mic(1,48000,16)

file_name = 'record_0.wav'
sensor.Record('./',file_name)

label = 'yes'
mfcc=False
sg = SignalGenerator(labels=LABELS, sampling_rate=16000, frame_length=640, frame_step=320,
            num_mel_bins=40, lower_frequency=20, upper_frequency=4000, num_coefficients=10, mfcc=mfcc)

test_ds = sg.make_dataset(file_name)

'''
input_data = tf.constant(x,dtype=tf.float32)
input_data = tf.expand_dims(input_data, 0)

interpreter = tf.lite.Interpreter(model_path='model_tflite')
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

interpreter.set_tensor(input_details[0]['index'], input_data)
interpreter.invoke()
my_output = interpreter.get_tensor(output_details[0]['index'])[0]

print(f'Measured: {y[0]}, {y[1]} --- Predicted: {my_output[0]:.2f}, {my_output[1]:.2f} --- MAE: {np.abs(y[0] - my_output[0]):.2f}, {np.abs(y[1] - my_output[1]):.2f}')

'''
