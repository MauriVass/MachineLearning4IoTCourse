import tensorflow as tf
import time
import os

input_file = 'image640x480.JPG'
image_input = tf.io.read_file(input_file)
image = tf.io.decode_jpeg(image_input)
print(f'Image size {os.path.getsize(input_file)}\n')

crop = 168
resize = 224
offset_height = (image.shape[0] - crop) // 2
offset_width = (image.shape[1] - crop) // 2
image = tf.image.crop_to_bounding_box(image,offset_height,offset_width,crop,crop)

methods = ['bilinear','bicubic','nearest','area','gaussian']
for m in methods:
	print(f'Method: {m}')
	start_time = time.time()
	image = tf.image.resize(image,(resize,resize),method=m)
	end_time = time.time()
	execution_time = end_time - start_time
	print(f'Execution Time {execution_time:.3f}')

	image = tf.cast(image,tf.uint8)
	image_output = tf.io.encode_jpeg(image)
	output_file = f'output_{m}_Ex4.JPG'
	tf.io.write_file(output_file,image_output)
	print(f'Image size {os.path.getsize(output_file)}\n')
