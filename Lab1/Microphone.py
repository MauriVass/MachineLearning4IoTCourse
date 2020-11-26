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

class Mic:
	def __init__(self,dur,rate,res,output):
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
		self.stream.wait()

		self.output_file = output

	def Record(self):
		print('Recording')
		frames = []

		max_val = int(self.rate / self.chunk * self.record_seconds)
		time_start = time.time()
		for i in range(max_val):
			data = stream.read(self.chunk, exception_on_overflow=False)
			frames.append(data)

		#Close streaming and destroy audio obj to free the memory
		stream.stop_stream()

		time_end = time.time()
		elapsed_time = time_end - time_start
		print(f'Finished Recording. Elapsed time {elapsed_time:.3f}')

		time_start = time.time()
		#Save file as a binary file
		waveFile = wave.open(self.output_file,'wb')
		waveFile.setnchannels(self.channels)
		waveFile.setsampwidth(audio.get_sample_size(self.resolution))
		waveFile.setframerate(self.rate)
		#Merge all the recorded frames as binary
		waveFile.writeframes(b''.join(frames))
		waveFile.close()
		time_end = time.time()

		storage_time = time_end - time_start
		print(f'Time to storage {storage_time:.3f}')

		#Print the file size value: higher resolution means higher file size
		print(f'File Size {os.path.getsize(self.output_file)/10**3}KB')

		def CloseBuffer(self):
			self.stream.close()
			self.audio.terminate()

if __name__=='__main__':
	#Create a parser obj to get input parameters
	parser = argparse.ArgumentParser()
	parser.add_argument('--duration', type=int, help='duration in s', required=True)
	parser.add_argument('--rate', type=int, help='sampling rate in Hz (check your microphone specific)', required=True)
	parser.add_argument('--resolution', type=int, help='resolution=(8, 16, 32)', required=True)
	parser.add_argument('-o', type=str, help='output', required=True)
	args = parser.parse_args()

        #Save the input parameters in variables
	duration = args.duration
	rate = args.rate
	resolution = args.resolution
	file_name = args.o

	#Create microphone instance
	microphone = Mic(duration,rate,resolution,file_name)
	#Start recording
	microphone.Record()
	#Close Stream
	microphone.CloseBuffer()

