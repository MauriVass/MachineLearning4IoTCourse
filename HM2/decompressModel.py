import zlib
import sys
import os

#files = [f for f in os.listdir('Models')]
#for f in files:
	#model_path =  f'Models/{f}'
model_path =  sys.argv[1]
with open(model_path, 'rb') as fp:
	model = zlib.decompress(fp.read())
	output_model = model_path[:-4]
	file = open(output_model,'wb')
	print('Saving: ',output_model)
	file.write(model)
	file.close()
