'''
Python script that precrocess an image given offset and dimension.
Input:
-The program requires 2 parameters:
	-an input .jpg image,
	-the crop size,
	-the resize size,
	-the name of the output file
Output:
-A .png file
'''

import tensorflow as tf
import time
import os
import argparse

class Preprocess:
	def __init__(self):
		self.methods = ['bilinear','bicubic','nearest','area','gaussian']

		def Calculate(self,image_input,crop,resize,output):
			input_file = image_input
			image_input = tf.io.read_file(input_file)
			image = tf.io.decode_jpeg(image_input)
			print(f'Image size {os.path.getsize(input_file)}\n')

			offset_height = (image.shape[0] - crop) // 2
			offset_width = (image.shape[1] - crop) // 2
			image = tf.image.crop_to_bounding_box(image,offset_height,offset_width,crop,crop)

			for m in self.methods:
				print(f'Method: {m}')
				start_time = time.time()
				image = tf.image.resize(image,(resize,resize),method=m)
				end_time = time.time()
				execution_time = end_time - start_time
				print(f'Execution Time {execution_time:.3f}')

				image = tf.cast(image,tf.uint8)
				image_output = tf.io.encode_jpeg(image)
				output_file = f'{output}_{m}.JPG'
				tf.io.write_file(output_file,image_output)
				print(f'Image size {os.path.getsize(output_file)}\n')

if __name__=='__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('-i',type=str, help='input spectrogram file', required=True)
	parser.add_argument('--crop',type=int, help='crop size', required=True)
	parser.add_argument('--resize',type=int, help='resize size', required=True)
	parser.add_argument('-o',type=str, help='output file', required=True)
	args = parser.parse_args()

	input = args.i
	crop = args.crop
	resize = args.resize
	output = args.o

	preprocc = Preprocess()
	preprocess.Calculate(input,crop,resize,output)
