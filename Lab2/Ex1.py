'''
Python script which resample an audio file from an high frequency to a lower one.
This is useful to reduce the file size without losing to much information.
Input:
-The program requires 1 parameters: an input .wav file
Output:
-A .wav file
'''

from scipy.io import wavfile
from scipy import signal
import numpy as np
import os
from playsound import playsound
import simpleaudio as sa
import argparse

#Receive input parameter
parser = argparse.ArgumentParser()
parser.add_argument('-i', type=str, help='input .wav file', required=True)
args = parser.parse_args()

def play_sound(path):
	print(f'Playing {path}')
	wave_obj = sa.WaveObject.from_wave_file(path)
	play_obj = wave_obj.play()
	play_obj.wait_done()

input_file = args.i
rate, audio = wavfile.read(input_file)

#Print the value of the input file
initial_size = os.path.getsize(input_file)
print(f'Initial size {initial_size}')

#Frequency used to resample the audio file.
#This value should a number such that the value ratio=init_freq/sampling_freq is integer.
#In this case the init_frq is 48k and sampling_rate is 16k
sampling_freq = 16000
ratio = rate / sampling_freq
audio = signal.resample_poly(audio,1,ratio)

audio = audio.astype(np.int16)

#Save the resampled audio
output_file = 'output.wav'
wavfile.write(output_file,sampling_freq,audio)

#Print the value of the output file (should be lower the the input size)
final_size = os.path.getsize(output_file)
print(f'Initial size {final_size}')

#Play both file to check if they (almost) are the same
print('Playing..')
play_sound(input_file)
play_sound(output_file)
