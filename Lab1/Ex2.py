'''
Python script which records an audio file using a Raspberry Pi module and a microphone.
Wiring:
-Connect the mic to an usb port
Input:
-The program requires 3 parameters: the sampling rate (in Hz), the resolution, the name of the output file
Output:
-A .wav file
'''

import pyaudio
import wave
import os
import argparse
import time

#Create a parser obj to get input parameters
parser = argparse.ArgumentParser()
parser.add_argument('--rate', type=int, help='sampling rate in Hz', required=True)
parser.add_argument('--resolution', type=int, help='resolution=(8, 16, 32)', required=True)
parser.add_argument('-o', type=str, help='output', required=True)
args = parser.parse_args()

#Get the pre-set resolution values using pyaudio library
input_res = args.resolution
if(input_res==8):
	resolution = pyaudio.paInt8
elif(input_res==16):
	resolution = pyaudio.paInt16
elif(input_res==32):
	resolution = pyaudio.paInt32
else:
	raise ValueError

#Set variables for recording
channels = 1
rate = args.rate
chunk = 4800
record_seconds = 1
output_file = args.o

audio = pyaudio.PyAudio()

stream = audio.open(format=resolution,
	channels=channels,
	rate= rate, input=True,
	frames_per_buffer=chunk)
print('Recording')
frames = []

max_val = int(rate / chunk * record_seconds)
time_start = time.time()
for i in range(max_val):
	data = stream.read(chunk, exception_on_overflow=False)
	frames.append(data)

time_end = time.time()
elapsed_time = time_end - time_start
print(f'Finished Recording. Elapsed time {elapsed_time:.3f}')

#Close streaming and destroy audio obj to free the memory
stream.stop_stream()
stream.close()
audio.terminate()

time_start = time.time()
#Save file as a binary file
waveFile = wave.open(output_file,'wb')
waveFile.setnchannels(channels)
waveFile.setsampwidth(audio.get_sample_size(resolution))
waveFile.setframerate(rate)
#Merge all the recorded frames as binary
waveFile.writeframes(b''.join(frames))
waveFile.close()
time_end = time.time()

storage_time = time_end - time_start
print(f'Time to storage {storage_time:.3f}')

#Print the file size value: higher resolution means higher file size
print(f'File Size {os.path.getsize(output_file)}')
