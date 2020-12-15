

import argparse
import numpy as np
import os
import pandas as pd
import tensorflow as tf
from tensorflow import keras

#parser = argparse.ArgumentParser()
#parser.add_argument('--model', type=str, required=False, help='model name: MLP, CNN, LSTM', default='MLP')
#parser.add_argument('--labels', type=int, required=False, help='model output', default=0)
#args = parser.parse_args()

#Set a seed to get repricable results
seed = 42
tf.random.set_seed(seed)
np.random.seed(seed)

#Download and extract the .csv file. The result is cached to avoid to download everytime
zip_path = tf.keras.utils.get_file(
    origin='https://storage.googleapis.com/tensorflow/tf-keras-datasets/jena_climate_2009_2016.csv.zip',
    fname='jena_climate_2009_2016.csv.zip',
    extract=True,
    cache_dir='.', cache_subdir='data')
csv_path, _ = os.path.splitext(zip_path)
df = pd.read_csv(csv_path)

#Take the required columns
column_indices = [2, 5]
columns = df.columns[column_indices]
data = df[columns].values.astype(np.float32)

#Separate the data in train, validation and test sets
n = len(data)
train_data = data[0:int(n*0.7)]
val_data = data[int(n*0.7):int(n*0.9)]
test_data = data[int(n*0.9):]
print(f'Total length: {n}, Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}')

class WindowGenerator:
	def __init__(self, mean, std):
		self.input_width = 6
		self.output_width = 6
		self.label_options = 2
		self.mean = tf.reshape(tf.convert_to_tensor(mean), [1, 1, 2])
		self.std = tf.reshape(tf.convert_to_tensor(std), [1, 1, 2])

	def split_window(self, features):
		print(1,features.shape)
		inputs = features[:, :-self.output_width, :]
		print(2,inputs.shape)

		multi_step = True
		if(multi_step is False):
			labels = features[:, -self.output_width, :]
			print(3,labels.shape)
			labels.set_shape([None, self.label_options])
			print(5,labels.shape,'\n\n')
		else:
			labels = features[:, -self.output_width:, :]
			print(3,labels.shape)
			labels.set_shape([None, self.output_width, self.label_options])
			print(5,labels.shape,'\n\n')

		#labels = tf.expand_dims(labels, -1)

		inputs.set_shape([None, self.input_width, self.label_options])
		print(4,inputs.shape)

		return inputs, labels

	def normalize(self, features):
		features = (features - self.mean) / (self.std + 1.e-6)
		return features

	def preprocess(self, features):
		inputs, labels = self.split_window(features)
		inputs = self.normalize(inputs)

		return inputs, labels

	def make_dataset(self, data, train):
		#The targets is None since the labels are already inside the data
		ds = tf.keras.preprocessing.timeseries_dataset_from_array(
						data=data,
						targets=None,
						sequence_length=self.input_width+self.output_width,
						sequence_stride=1,
						batch_size=32)
		ds = ds.map(self.preprocess)
		ds = ds.cache()
		if train is True:
			ds = ds.shuffle(100, reshuffle_each_iteration=True)

		return ds


#Calculate statistics for normalization
mean = train_data.mean(axis=0)
std = train_data.std(axis=0)

generator = WindowGenerator(mean, std)
train_ds = generator.make_dataset(train_data, True)
val_ds = generator.make_dataset(val_data, False)
test_ds = generator.make_dataset(test_data, False)
print(f'Train: {len(train_ds)}, Val: {len(val_ds)}, Test: {len(test_ds)}')


import time
if(True):
	for x,y in train_ds.take(1):
		print(x.shape) #(32,6,2)
		print(y.shape) #(32,6,1 or 2)

#print(model.model.metrics_names)

#pip install tensorflow_model_optimization
#import tensorflow_model_optimization as tfmot


class TempHumMAE(keras.metrics.Metric):
	def __init__(self, name='mean_absolute_error_cust', **kwargs):
		super().__init__(name, **kwargs)
		#initialiaze the variables used to calculate the loss
		self.count = self.add_weight(name='count', initializer='zeros')
		#The shape [2]('shape=(2,)' is equivalent) is for temperature ad humidity
		self.total = self.add_weight(name='total', initializer='zeros', shape=(2,))

	#Called at every batch of data
	def update_state(self, y_true, y_pred, sample_weight=None):
		#print('Prediction',y_pred)
		#print('True',y_true)
		error = tf.abs(y_pred-y_true)
		error = tf.reduce_mean(error, axis=(0,1))#
		#print(error)
		#You can just use + sign but it is better to use assign_add method
		self.total.assign_add(error)
		self.count.assign_add(1.)
		return
	def reset_states(self):
		self.count.assign(tf.zeros_like(self.count))
		self.total.assign(tf.zeros_like(self.total))
		return
	def result(self):
		results = tf.math.divide_no_nan(self.total, self.count)
		return results

def my_schedule(epoch, lr):
	if epoch < 10:
		return lr
	else:
		return lr * tf.math.exp(-0.1)

#https://www.tensorflow.org/tutorials/structured_data/time_series#single_step_models
class Model:
	def __init__(self,model_type,alpha=1,sparsity=None):
		self.alpha = alpha
		self.label=2
		self.n_output = 1 if self.label < 2 else 2
		self.metric = ['mae'] if self.label < 2 else [TempHumMAE()]
		self.model_type = model_type
		if(model_type=='MLP'):
			self.model = self.MLPmodel()
		elif(model_type=='CNN'):
			self.model = self.CNNmodel(alpha)
		elif(model_type=='LSTM'):
			self.model = self.LSTMmodel(alpha)

		#CALLBACKS
		self.callbacks = []
		self.checkpoint_path = 'THckp/'
		monitor_loss = 'mean_squared_error'
		self.model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
		filepath=self.checkpoint_path,
		save_weights_only=True,
		monitor=monitor_loss,
		mode='auto',
		save_best_only=True)
		self.callbacks.append(self.model_checkpoint_callback)

		self.early_stopping = tf.keras.callbacks.EarlyStopping(
		monitor=monitor_loss, min_delta=0.05, patience=3, verbose=1, mode='auto',
		baseline=None, restore_best_weights=True)
		self.callbacks.append(self.early_stopping)

		self.lr_exp = tf.keras.callbacks.LearningRateScheduler(my_schedule, verbose=1)
		#self.callbacks.append(self.lr_exp)
		self.lr_onplateau = tf.keras.callbacks.ReduceLROnPlateau(monitor=monitor_loss, factor=0.1,
		patience=2, min_lr=0.001, verbose=1)
		self.callbacks.append(self.lr_onplateau)

		self.sparsity = sparsity

		self.model.compile(optimizer='adam', loss=tf.keras.losses.MeanSquaredError(),
                  metrics=[self.metric,tf.keras.losses.MeanSquaredError()])
   
  def MLPmodel(self):
    model = keras.Sequential([
          keras.layers.Flatten(),
          keras.layers.Dense(4, activation='relu'),
          keras.layers.Dense(2, activation='relu'),
          keras.layers.Dense(self.n_output*6),
          keras.layers.Reshape([6, 2])
        ])
    return model
  def CNNmodel(self,alpha):
    model = keras.Sequential([
          keras.layers.Conv1D(filters=int(64*2*self.alpha),kernel_size=(3,),activation='relu'),
          keras.layers.Flatten(),
          keras.layers.Dense(int(64*self.alpha/2), activation='relu'),
          keras.layers.Dense(self.n_output*6),
          keras.layers.Reshape([6, 2])
        ])
    return model
  def LSTMmodel(self,alpha):
    model = keras.Sequential([
          keras.layers.LSTM(units=int(64*alpha)),
          keras.layers.Flatten(),
          keras.layers.Dense(self.n_output),
          keras.layers.Reshape([6, 2])
        ])
    return model

  def Train(self,train,validation,epoch):
    if(False):
      for c in self.callbacks:
        print(c)
    history = self.model.fit(train, batch_size=32, epochs=epoch, verbose=1,
                        validation_data=validation, validation_freq=2, callbacks=self.callbacks)#
    return history

  def Test(self, test, best=False):
    if(best):
      self.model.load_weights(self.checkpoint_path)
    error = self.model.evaluate(test, verbose=1)
    return error[1]

  def SaveModel(self,output,best=False):
      output += self.model_type
      if(self.alpha!=1):
        output += f'alpha{self.alpha}'
      if(self.sparsity!=None):
        output += f'spars{self.sparsity}'
      if(best):
        self.model.load_weights(self.checkpoint_path)
      if(self.sparsity):
        self.Strip()
      run_model = tf.function(lambda x: self.model(x))
      concrete_func = run_model.get_concrete_function(tf.TensorSpec([1,6,2], tf.float32))
      output = output.replace('.','_')
      print(f'Saving: {output}')
      self.model.save(output, signatures=concrete_func)
      #self.model.save(output)

  def Strip(self):
    self.model = tfmot.sparsity.keras.strip_pruning(self.model)



#Sparcity increases latency(may be a problem for KS) due to cache misses
#it can be (0,1) or None
sparsity = 0.9
pruning_params = {
	'pruning_schedule':
		tfmot.sparsity.keras.PolynomialDecay(
		initial_sparsity=0.30,
		final_sparsity=sparsity,
		begin_step=len(train_ds)*3,
		end_step=len(train_ds)*15)}
  
# It can be ['MLP', 'CNN', 'LSTM']
model_type = 'MLP'

#It can be (0,1]
alpha = 1

model = Model(model_type,alpha=alpha,sparsity=None)
init = time.time()
hist = model.Train(train_ds, val_ds, 20)
end = time.time()
print(f'{end-init}')

best = True
if(best is False):
	error = model.Test(test_ds)
	temp_loss, hum_loss = error
	print(f'Loss: Temp={temp_loss}, Hum={hum_loss}')
	model.SaveModel(f'TH')
	print('\n')
else:
	error = model.Test(test_ds,best)
	temp_loss, hum_loss = error
	print(f'Loss: Temp={temp_loss}, Hum={hum_loss}')
	model.SaveModel(f'TH_',True)


print(model.model.summary())

#!rm -r THckp/

#Save the tensor so that you can use them in other scripts
#tf.data.experimental.save(train_ds, './th_train')
#tf.data.experimental.save(val_ds, './th_val')
#tf.data.experimental.save(test_ds, './th_test')

#tensor_specs = (tf.TensorSpec([None, 6, 2], dtype=tf.float32),tf.TensorSpec([None, 6, 2]))
#train_ds = tf.data.experimental.load('./th_train', tensor_specs)
#val_ds = tf.data.experimental.load('./th_val', tensor_specs)
#test_ds = tf.data.experimental.load('./th_test', tensor_specs)

'''
#Deployer, Optimizer W_WA
import argparse
import tensorflow as tf
import os

def representative_dataset_gen():
    for x, _ in train_ds.take(1000):
        yield [x]

def Optimize(saved_model_dir,quantization,zipping):
  converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
  if(quantization=='w'):
      print('Only Weight')
      #Quantization Weights only
      converter.optimizations = [tf.lite.Optimize.DEFAULT]
      mini_float = False
      if(mini_float):
          converter.target_spec.supported_types = [tf.float16]
      tflite_model_dir = saved_model_dir + '.tflite_W'
  elif(quantization=='wa'):
      print('Weight Activation')
      #Quantization Weights and Activation
      converter.optimizations = [tf.lite.Optimize.DEFAULT]
      converter.representative_dataset = representative_dataset_gen
      tflite_model_dir = saved_model_dir + '.tflite_WA'
  else:
      tflite_model_dir = saved_model_dir + '.tflite'
  tflite_model = converter.convert()

  #Compression
  if(zipping is False):
      with open(tflite_model_dir, 'wb') as fp:
          fp.write(tflite_model)
  else:
      print('Compression')
      import zlib
      tflite_model_dir = tflite_model_dir + '.zip'
      with open(tflite_model_dir, 'wb') as fp:
          tflite_compressed = zlib.compress(tflite_model)#,level=9
          fp.write(tflite_compressed)

  print('Saving: ', tflite_model_dir)
  size_tflite_model = os.path.getsize(tflite_model_dir)
  print(f'Tflite Model size: {(size_tflite_model/1024):.2f} kB')

#Optimization for TH Forecasting
#any -> none
#w -> only weights
#wa -> weights and activation (have some problem with the shape/last reshape layer (maybe))
quantization = 'wq' 
zipping = False
saved_model_dir = 'TH_MLP/DSCNN'
Optimize(saved_model_dir,quantization,zipping)

#Decompress
import zlib
model_path = 'THmodelMLP.tflite.zip' 
with open(model_path, 'rb') as fp:
    model = zlib.decompress(fp.read())
    output_model = model_path[:-4]
    file = open(output_model,'wb')
    print('Saving: ',output_model)
    file.write(model)
    file.close()

#Test Models
import time
import tensorflow.lite as tflite

saved_model_dir='THmodelMLP.tflite'

test_ds = test_ds.unbatch().batch(1)

interpreter = tf.lite.Interpreter(model_path=saved_model_dir)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
mae = [0,0]
n = 0
time_infe = 0
print(test_ds)

for x,y in test_ds:
  #print(x,y)
  input_data = x
  y_true = y.numpy()[0]
  
  ti = time.time()
  interpreter.set_tensor(input_details[0]['index'], input_data)
  interpreter.invoke()
  my_output = interpreter.get_tensor(output_details[0]['index'])[0]
  time_infe += time.time()-ti

  n+=1
  #mae[0] += np.abs(y[0] - my_output[0])
  #mae[1] += np.abs(y[1] - my_output[1])
  error = tf.abs(my_output-y_true)
  mae += tf.reduce_mean(error, axis=(0,))

  #print(f'Measured: {y[0]}, {y[1]} --- Predicted: {my_output[0]:.2f}, {my_output[1]:.2f} --- MAE: {:.2f}, {np.abs(y[1] - my_output[1]):.2f}')
print(f'MAE: temp: {mae[0]/n}, humi: {mae[1]/n}, time: {(time_infe/n)*1000} ms')

#          Destination      Origin
!zip -r ./th_test.zip ./th_test1

!unzip THFmodelCNN.zip ./THFmodelCNN

#Keyword Spotting
import argparse
import numpy as np
import os
import tensorflow as tf
from tensorflow import keras
!pip install tensorflow_model_optimization
import tensorflow_model_optimization as tfmot

#Set a seed to get repricable results
seed = 42
tf.random.set_seed(seed)
np.random.seed(seed)

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

	def make_dataset(self, files, train=False):
		#This method creates a dataset from a numpy array (our listfile path)
		ds = tf.data.Dataset.from_tensor_slices(files)
		#Different preprocess step depending on the input parameter
		ds = ds.map(self.preprocess, num_parallel_calls=4)
		ds = ds.batch(32)
		ds = ds.cache()

		if(train is True):
			ds = ds.shuffle(100, reshuffle_each_iteration=True)
		return ds

#Sparcity increases latency due to cache misses
class Model:
  def __init__(self,model_type,mfcc,train_ds,alpha=1,sparsity=None):
    self.alpha = alpha
    self.sparsity=sparsity
    self.n_output = 8
    if(mfcc):
      self.strides = [2,1]
    else:
      self.strides = [2,2]

    self.model_type = model_type
    if(model_type=='MLP'):
      self.model = self.MLPmodel()
    elif(model_type=='CNN'):
      self.model = self.CNNmodel()
    elif(model_type=='DSCNN'):
      self.model = self.DSCNNmodel()

    self.mfcc = mfcc

    #CALLBACKS
    self.callbacks = []
    self.checkpoint_path = 'KSckp/'
    monitor = 'val_sparse_categorical_accuracy'
    self.model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
      filepath=self.checkpoint_path,
      save_weights_only=True,
      monitor=monitor,
      mode='max',
      save_best_only=True)
    self.callbacks.append(self.model_checkpoint_callback)

    self.early_stopping = tf.keras.callbacks.EarlyStopping(
      monitor=monitor, min_delta=0, patience=4, verbose=1, mode='auto',
      baseline=None, restore_best_weights=True)
    self.callbacks.append(self.early_stopping)

    #self.lr_exp = tf.keras.callbacks.LearningRateScheduler(my_schedule, verbose=1)
    #self.callbacks.append(self.lr_exp)
    self.lr_onplateau = tf.keras.callbacks.ReduceLROnPlateau(monitor=monitor, factor=0.1,
      patience=2, min_lr=0.001, verbose=1)
    #self.callbacks.append(self.lr_onplateau)

    self.sparsity = sparsity
    if(self.sparsity is not None):
      pruning_params = {
        'pruning_schedule':
        tfmot.sparsity.keras.PolynomialDecay(
        initial_sparsity=0.30,
        final_sparsity=sparsity,
        begin_step=len(train_ds)*3,
        end_step=len(train_ds)*15)}

      prune_low_magnitude = tfmot.sparsity.keras.prune_low_magnitude
      self.model = prune_low_magnitude(self.model, **pruning_params)
      self.model_sparcity_callback = tfmot.sparsity.keras.UpdatePruningStep()
      self.callbacks.append(self.model_sparcity_callback)
      self.callbacks.append(tfmot.sparsity.keras.PruningSummaries(log_dir='PruningSumm/'))
      if(mfcc):
        input_shape = [None, 49, 10, 1]
      else:
        input_shape = [None, 32, 32, 1]
      self.model.build(input_shape)

    self.model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                       metrics=['sparse_categorical_accuracy'])

  def MLPmodel(self):
    model = keras.Sequential([
      keras.layers.Flatten(),
      keras.layers.Dense(int(256*self.alpha), activation='relu'),
      keras.layers.Dense(int(256*self.alpha), activation='relu'),
      keras.layers.Dense(int(256*self.alpha), activation='relu'),
      keras.layers.Dense(self.n_output)
      ])
    return model

	#Strides = [2,2] if STFT, [2,1] if MFCC
  def CNNmodel(self):
    model = keras.Sequential([
      keras.layers.Conv2D(filters=int(128*self.alpha),kernel_size=[3,3],strides=self.strides,use_bias=False),
      keras.layers.BatchNormalization(momentum=0.1),
      keras.layers.Activation('relu'),
      keras.layers.Conv2D(filters=128,kernel_size=[3,3],strides=[1,1],use_bias=False),
      keras.layers.BatchNormalization(momentum=0.1),
      keras.layers.Activation('relu'),
      keras.layers.Conv2D(filters=128,kernel_size=[3,3],strides=[1,1],use_bias=False),
      keras.layers.BatchNormalization(momentum=0.1),
      keras.layers.Activation('relu'),
      keras.layers.GlobalAveragePooling2D(),
      keras.layers.Dense(self.n_output)
      ])
    return model

  def DSCNNmodel(self):
    model = keras.Sequential([
      keras.layers.Conv2D(filters=int(256*self.alpha),kernel_size=[3,3],strides=self.strides,use_bias=False), #input_shape=(32,32,1)
      keras.layers.BatchNormalization(momentum=0.1),
      keras.layers.Activation('relu'),
      keras.layers.DepthwiseConv2D(kernel_size=[3, 3], strides=[1, 1], use_bias=False),
      keras.layers.Conv2D(filters=256,kernel_size=[1,1],strides=[1,1],use_bias=False),
      keras.layers.BatchNormalization(momentum=0.1),
      keras.layers.Activation('relu'),
      keras.layers.DepthwiseConv2D(kernel_size=[3, 3], strides=[1, 1], use_bias=False),
      keras.layers.Conv2D(filters=256,kernel_size=[1,1],strides=[1, 1],use_bias=False),
      keras.layers.BatchNormalization(momentum=0.1),
      keras.layers.Activation('relu'),
      keras.layers.GlobalAveragePooling2D(),
      keras.layers.Dense(self.n_output)
      ])
    return model

  def Train(self,train,validation,epoch):
    print('Training')
    history = self.model.fit(train, batch_size=32, epochs=epoch, verbose=1,
        validation_data=validation, validation_freq=1, callbacks=self.callbacks)
    return history

  def Test(self, test, best=True):
    print('Evaluation')
    if(best):
        self.model.load_weights(self.checkpoint_path)
    loss, error = self.model.evaluate(test, verbose=1)
    return (loss, error)

  def SaveModel(self,output,best=True):
    output += self.model_type
    output += str(self.mfcc)
    if(self.alpha!=1):
      output += f'alpha{self.alpha}'
    if(self.sparsity!=None):
      output += f'spars{self.sparsity}'
    if(best):
      self.model.load_weights(self.checkpoint_path)
    if(self.sparsity):
      self.Strip()
    #run_model = tf.function(lambda x: self.model(x))
    #concrete_func = run_model.get_concrete_function(tf.TensorSpec([1,6,2], tf.float32))
    print(f'Saving: {output}')
    self.model.save(output)
    #self.model.save(output)

  def Strip(self):
    self.model = tfmot.sparsity.keras.strip_pruning(self.model)

#Download and extract the .csv file. The result is cached to avoid to download everytime
zip_path = tf.keras.utils.get_file(
	origin='http://storage.googleapis.com/download.tensorflow.org/data/mini_speech_commands.zip',
	fname='mini_speech_commands.zip',
	extract=True,
	cache_dir='.', cache_subdir='data')

data_dir = os.path.join('.', 'data', 'mini_speech_commands')
#Store in a list all files as a string inside a given path
#filenames = tf.io.gfile.glob(str(data_dir) + '/*/*')
#Shuffle to have a uniform distribution of the samples
#filenames = tf.random.shuffle(filenames)
#num_samples = len(filenames)
# train_files = filenames[:int(num_samples*0.8)]
# validation_files = filenames[int(num_samples*0.8):int(num_samples*0.9)]
# test_files = filenames[int(num_samples*0.9):]

def readFile(file):
  elems = []
  fp = open(file,'r')
  for f in fp:
    elems.append(f.strip())
  return elems

#Spit the dataset following the .pdf requirements
train_files = readFile('kws_train_split.txt')
validation_files = readFile('kws_val_split.txt')
test_files = readFile('kws_test_split.txt')

#Extract the labes: folders inside the data folder
LABELS = np.array(tf.io.gfile.listdir(str(data_dir)))
#Remove the 'README.md' file, since not useful
LABELS = LABELS[LABELS != 'README.md']


#for m in ['MLP', 'CNN', 'DSCNN']:
  #for f in [False, True]:
m = 'CNN'
f = True
print(f'\n\nModel: {m}, mfcc: {f}')
model = m
mfcc = f

if(mfcc):
	sg = SignalGenerator(labels=LABELS, sampling_rate=16000, frame_length=640, frame_step=320,
				num_mel_bins=40, lower_frequency=20, upper_frequency=4000, num_coefficients=10, mfcc=mfcc)
else:
	sg = SignalGenerator(labels=LABELS, sampling_rate=16000, frame_length=256, frame_step=128)

train_ds = sg.make_dataset(train_files,True)
val_ds = sg.make_dataset(validation_files)
test_ds = sg.make_dataset(test_files)
print(f'Train: {len(train_ds)}, Val: {len(val_ds)}, Test: {len(test_ds)}')

for x,y in train_ds.take(1):
	print(x.shape,y.shape)

save_best = False
model = Model(model,mfcc,train_ds,alpha=0.8,sparsity=0.9)
hist = model.Train(train_ds,val_ds,20)
loss, acc = model.Test(test_ds,save_best)
print('Accuracy test set: ',acc)
model.SaveModel(f'KS_',save_best)
model.model.summary()

print(model.model.metrics_names)

hist.history

loss, acc = model.Test(test_ds,save_best)
print('Accuracy test set: ',acc)

#any -> none
#w -> only weights
#wa -> weights and activation
quantization = 'wc' 
zipping = False
saved_model_dir = 'KS_CNNTruealpha0.8spars0.9'

Optimize(saved_model_dir,quantization,zipping)

#Test Models
import time
import tensorflow.lite as tflite

saved_model_dir='KS_CNNTruealpha0.8spars0.9.tflite'
test_ds1 = test_ds.unbatch().batch(1)

interpreter = tf.lite.Interpreter(model_path=saved_model_dir)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
acc = 0
n = 0
time_infe = 0
print(test_ds1)

for x,y in test_ds1:
  #print(x,y)
  input_data = x
  y_true = y.numpy()[0]
  
  ti = time.time()
  interpreter.set_tensor(input_details[0]['index'], input_data)
  interpreter.invoke()
  my_output = interpreter.get_tensor(output_details[0]['index'])[0]
  time_infe += time.time()-ti

  n+=1
  index_pred = np.argmax(my_output)
  if(index_pred):
    acc += 1

print(f'Accuracy: {(acc/n):.3f}, time: {(time_infe/n)*1000} ms')

#To run on board
import argparse
import numpy as np
from subprocess import call
import tensorflow as tf
import time
from scipy import signal

#Evaluation
model = 'DSCNN.tflite' #path tflite
rate = 16000
mfcc = False
if(mfcc):
  length = 640
  stride = 320
else:
  length = 256
  stride = 128
resize = 32
num_mel_bins = 40
num_coefficients = 10

num_frames = (rate - length) // stride + 1
num_spectrogram_bins = length // 2 + 1

linear_to_mel_weight_matrix = tf.signal.linear_to_mel_weight_matrix(
        num_mel_bins, num_spectrogram_bins, rate, 20, 4000)

if model is not None:
    interpreter = tf.lite.Interpreter(model_path=model)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()


inf_latency = []
tot_latency = []
for i in range(100):
    sample = np.array(np.random.random_sample(48000), dtype=np.float32)

    start = time.time()

    # Resampling
    sample = signal.resample_poly(sample, 1, 48000 // rate)

    sample = tf.convert_to_tensor(sample, dtype=tf.float32)

    # STFT
    stft = tf.signal.stft(sample, length, stride,
            fft_length=length)
    spectrogram = tf.abs(stft)

    if mfcc is False and resize > 0:
        # Resize (optional)
        spectrogram = tf.reshape(spectrogram, [1, num_frames, num_spectrogram_bins, 1])
        spectrogram = tf.image.resize(spectrogram, [resize, resize])
        input_tensor = spectrogram
    else:
        # MFCC (optional)
        mel_spectrogram = tf.tensordot(spectrogram, linear_to_mel_weight_matrix, 1)
        log_mel_spectrogram = tf.math.log(mel_spectrogram + 1.e-6)
        mfccs = tf.signal.mfccs_from_log_mel_spectrograms(log_mel_spectrogram)
        mfccs = mfccs[..., :num_coefficients]
        mfccs = tf.reshape(mfccs, [1, num_frames, num_coefficients, 1])
        input_tensor = mfccs

    if model is not None:
        interpreter.set_tensor(input_details[0]['index'], input_tensor)
        start_inf = time.time()
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]['index'])

    end = time.time()
    tot_latency.append(end - start)

    if model is None:
        start_inf = end

    inf_latency.append(end - start_inf)
    time.sleep(0.1)

print('Inference Latency {:.2f}ms'.format(np.mean(inf_latency)*1000.))
print('Total Latency {:.2f}ms'.format(np.mean(tot_latency)*1000.))

'''
