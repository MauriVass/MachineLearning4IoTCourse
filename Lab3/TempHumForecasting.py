import argparse
import numpy as np
import os
import pandas as pd
import tensorflow as tf
from tensorflow import keras

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, required=False, help='model name: MLP, CNN, LSTM', default='MLP')
parser.add_argument('--labels', type=int, required=False, help='model output', default=0)
args = parser.parse_args()

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
#Calculate statistics for normalization
mean = train_data.mean(axis=0)
std = train_data.std(axis=0)

#This can be 0,1,2: 0 only temp, 1 only humy, 2 prediction for both
input_width = 6
LABEL_OPTIONS = args.labels

class WindowGenerator:
	def __init__(self, input_width, label_options, mean, std):
		self.input_width = input_width
		self.label_options = label_options
		self.mean = tf.reshape(tf.convert_to_tensor(mean), [1, 1, 2])
		self.std = tf.reshape(tf.convert_to_tensor(std), [1, 1, 2])

	def split_window(self, features):
		inputs = features[:, :-1, :]

		if self.label_options < 2:
			labels = features[:, -1, self.label_options]
			labels = tf.expand_dims(labels, -1)
			num_labels = 1
		else:
			labels = features[:, -1, :]
			num_labels = 2

		inputs.set_shape([None, self.input_width, 2])
		labels.set_shape([None, num_labels])

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
						sequence_length=input_width+1,
						sequence_stride=1,
						batch_size=32)
		ds = ds.map(self.preprocess)
		ds = ds.cache()
		if train is True:
			ds = ds.shuffle(100, reshuffle_each_iteration=True)

		return ds

class TempHumMAE(keras.metrics.Metric):
	def __init__(self, name='mean_absolute_error', **kwargs):
		super().__init__(name, **kwargs)
		#initialiaze the variables used to calculate the loss
		self.count = self.add_weight(name='count', initializer='zeros')
		#The shape [2]('shape=(2,)' is equivalent) is for temperature ad humidity
		self.total = self.add_weight(name='total', initializer='zeros', shape=(2,))

	#Called at every batch of data
	def update_state(self, y_true, y_pred, sample_weight=None):
		error = tf.abs(y_pred-y_true)
		error = tf.reduce_mean(error, axis=0)
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

class Model:
	def __init__(self,label,model_type):
		self.n_output = 1 if label < 2 else 2
		self.metric = ['mae'] if label < 2 else [TempHumMAE()]
		if(model_type=='MLP'):
			self.model = self.MLPmodel()
		elif(model_type=='CNN'):
			self.model = self.CNNmodel()
		elif(model_type=='LSTM'):
			self.model = self.LSTMmodel()
		self.model.compile(optimizer='adam', loss=tf.keras.losses.MeanSquaredError(), metrics=self.metric)

	def MLPmodel(self):
		model = keras.Sequential([
					keras.layers.Flatten(),
					keras.layers.Dense(128, activation='relu'),
					keras.layers.Dense(128, activation='relu'),
					keras.layers.Dense(self.n_output),
				])
		return model
	def CNNmodel(self):
		model = keras.Sequential([
					keras.layers.Conv1D(filters=64,kernel_size=(3,),activation='relu'),
					keras.layers.Flatten(input_shape=(64,2)),
					keras.layers.Dense(64, activation='relu'),
					keras.layers.Dense(self.n_output),
				])
		return model
	def LSTMmodel(self):
		model = keras.Sequential([
					keras.layers.LSTM(units=64),
					keras.layers.Flatten(input_shape=(64,2)),
					keras.layers.Dense(self.n_output),
				])
		return model

	def Train(self,train,validation,epoch):
		history = self.model.fit(train, batch_size=32, epochs=epoch, verbose=1, validation_data=validation, validation_steps=10)
		return history

	def Test(self, test):
		loss, error = self.model.evaluate(test, verbose=1)
		return (loss, error)

	def SaveModel(self,output):
		run_model = tf.function(lambda x: self.model(x))
		concrete_func = run_model.get_concrete_function(tf.TensorSpec([1,6,2], tf.float32))
		self.model.save(output, signatures=concrete_func)

#MLP, CNN-1D, LSTM
model_type = args.model
generator = WindowGenerator(input_width, LABEL_OPTIONS, mean, std)
train_ds = generator.make_dataset(train_data, True)
val_ds = generator.make_dataset(val_data, False)
test_ds = generator.make_dataset(test_data, False)
print(f'Train: {len(train_ds)}, Val: {len(val_ds)}, Test: {len(test_ds)}')

if(False):
	for x,y in train_ds.take(1):
		print(x.shape,x) #(32,6,2)
		print(y.shape,y) #(32,6,1 or 2)

model = Model(LABEL_OPTIONS,model_type)
hist = model.Train(train_ds, val_ds, 20)
loss, error = model.Test(test_ds)

if(LABEL_OPTIONS<2):
	temp_loss = error
	print(f'Loss: Temp={temp_loss}')
else:
	temp_loss, hum_loss = error
	print(f'Loss: Temp={temp_loss}, Hum={hum_loss}')
print(model.model.summary())
model.SaveModel(f'models/model_{model_type}/')
