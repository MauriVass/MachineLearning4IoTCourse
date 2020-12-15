'''
Python script which records an audio file using a Raspberry Pi module and a microphone.
Wiring:
-Connect the mic to an usb port
Input:
-The program requires 4 parameters:
	--the duration of the registration (in s)
	--the sampling rate (in Hz)
	--the resolution (8, 16, 32)
	--the name of the output file
Output:
-A .wav file
'''

import pyaudio
import wave
import os
import argparse
import time
from Resampler import Resampler

class Mic:
	def __init__(self,dur,rate,res):
		#Get the pre-set resolution values using pyaudio library
		self.input_res = int(res)
		if(self.input_res==8):
			self.resolution = pyaudio.paInt8
		elif(self.input_res==16):
			self.resolution = pyaudio.paInt16
		elif(self.input_res==32):
			self.resolution = pyaudio.paInt32
		else:
			raise ValueError

		#Set variables for recording
		self.channels = 1
		self.rate = rate
		#This is the amount of memory that can be stored on the ram while recording
		#After this memory is full, it will be freed storing the samples on the HD
		self.chunk = 4800
		self.record_seconds = dur

		self.audio = pyaudio.PyAudio()

		self.stream = self.audio.open(format=self.resolution,
			channels=self.channels,
			rate= self.rate, input=True,
			frames_per_buffer=self.chunk)

	def Record(self,output_file,i):
		print('Recording ',i)
		frames = []

		max_val = int(self.rate / self.chunk * self.record_seconds)
		time_start = time.time()

		self.stream.start_stream()
		for i in range(max_val):
			data = self.stream.read(self.chunk, exception_on_overflow=False)
			frames.append(data)
		#Close streaming and destroy audio obj to free the memory
		self.stream.stop_stream()

		time_end = time.time()
		elapsed_time = time_end - time_start
		print(f'Finished Recording. Elapsed time {elapsed_time:.3f}')

		#time_start = time.time()
		#Save file as a binary file
		waveFile = wave.open(output_file,'wb')
		waveFile.setnchannels(self.channels)
		waveFile.setsampwidth(self.audio.get_sample_size(self.resolution))
		waveFile.setframerate(self.rate)
		#Merge all the recorded frames as binary
		waveFile.writeframes(b''.join(frames))
		waveFile.close()
		#time_end = time.time()

		#storage_time = time_end - time_start
		#print(f'Time to storage {storage_time:.3f}')

		#Print the file size value: higher resolution means higher file size
		#print(f'File Size {os.path.getsize(output_file)/10**3}KB')

	def CloseBuffer(self):
		self.stream.close()
		self.audio.terminate()

if __name__=='__main__':
	#Create a parser obj to get input parameters
	#Save the input parameters in variables
	duration = 1
	rate = 48000
	resolution = 16

	resampler = Resampler()

	#Create microphone instance
	microphone = Mic(duration,rate,resolution)
	folder = 'data/mini_speech_commands/silence/'
	tmp = 'tmp'
	for i in range(1000):
		#Start recording
		file_name = f'{folder}silence_{i}.wav'
		microphone.Record(folder+tmp,i)
		resampler.Resample(folder+tmp,file_name)
		os.remove(folder+tmp)
	#Close Stream
microphone.CloseBuffer()

