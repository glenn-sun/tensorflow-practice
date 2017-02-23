# Modified from tf.contrib.learn.python.learn.datasets.mnist.py
# and from tf.contrib.learn.python.learn.datasets.base.py, which are
# licensed under the Apache License, Version 2.0. A copy of this
# license can be found at http://www.apache.org/licenses/LICENSE-2.0
# ==================================================================

import numpy as np
import os
import random
import urllib
import cPickle
import collections
import tarfile

from tensorflow.python.platform import gfile

class DataSet(object):

	def __init__(self, data, labels, augment_image=False):
		"""Construct a DataSet."""

		assert data.shape[0] == labels.shape[0], ('data.shape: %s labels.shape: %s' 
												  % (data.shape, labels.shape))
		self._num_examples = data.shape[0]
		
		# Convert from [0, 255] -> [0.0, 1.0].
		data = data.astype(np.float32)
		data = np.multiply(data, 1.0 / 255.0)

		self._data = data
		self._labels = labels
		self._epochs_completed = 0
		self._index_in_epoch = 0
		self._augment_image = augment_image
		self._freeze = False

	@property
	def data(self):
		return self._data

	@property
	def labels(self):
		return self._labels

	@property
	def num_examples(self):
		return self._num_examples

	@property
	def epochs_completed(self):
		return self._epochs_completed

	def toggle_freeze(self):
		self._freeze = not self._freeze

	def augment_image(self, data_batch, batch_size):
		new_batch = np.empty([batch_size, 24, 24, 3])
		for img in range(batch_size):
			c_offset, r_offset = np.random.randint(0, 8, size=2)
			new_batch[img] = data_batch[img, c_offset:c_offset + 24, r_offset:r_offset + 24, :]
		flip = np.random.choice([True, False], size=[batch_size, 1, 1, 1])
		new_batch = np.logical_not(flip) * new_batch + flip * new_batch[:, :, ::-1, :]
		brightness_delta = np.random.uniform(-0.125, 0.125, size=[batch_size, 1, 1, 1])
		new_batch = np.clip(new_batch + brightness_delta, 0, 1)
		return new_batch

	def test_augment_image(self, data_batch, batch_size):
		new_batch = np.empty([batch_size, 24, 24, 3])
		for img in range(batch_size):
			new_batch[img] = data_batch[img, 4:28, 4:28, :]
		return new_batch

	def next_batch(self, batch_size, test_time=False):
		"""Return the next `batch_size` examples from this data set."""
		start = self._index_in_epoch
		# Shuffle for the first epoch
		if self._epochs_completed == 0 and start == 0:
			if not self._freeze:
				perm0 = np.arange(self._num_examples)
				np.random.shuffle(perm0)
				self._data = self.data[perm0]
				self._labels = self.labels[perm0]
		# Go to the next epoch
		if start + batch_size > self._num_examples:
			# Finished epoch
			self._epochs_completed += 1
			# Get the rest examples in this epoch
			rest_num_examples = self._num_examples - start
			data_rest_part = self._data[start:self._num_examples]
			labels_rest_part = self._labels[start:self._num_examples]
			# Shuffle the data
			if not self._freeze:
				perm = np.arange(self._num_examples)
				np.random.shuffle(perm)
				self._data = self.data[perm]
				self._labels = self.labels[perm]
			# Start next epoch
			start = 0
			self._index_in_epoch = batch_size - rest_num_examples
			end = self._index_in_epoch
			data_new_part = self._data[start:end]
			labels_new_part = self._labels[start:end]
			if self._augment_image and test_time:
				return (self.test_augment_image(np.concatenate((data_rest_part, data_new_part), 
						axis=0), batch_size), np.concatenate((labels_rest_part, labels_new_part), 
						axis=0))
			elif self._augment_image:
				return (self.augment_image(np.concatenate((data_rest_part, data_new_part), axis=0),
						batch_size), np.concatenate((labels_rest_part, labels_new_part), axis=0))
			else:
				return (np.concatenate((data_rest_part, data_new_part), axis=0),
					    np.concatenate((labels_rest_part, labels_new_part), axis=0))
		else:
			self._index_in_epoch += batch_size
			end = self._index_in_epoch
			if self._augment_image and test_time:
				return (self.test_augment_image(self._data[start:end], batch_size), 
						self._labels[start:end])
			elif self._augment_image:
				return (self.augment_image(self._data[start:end], batch_size), 
						self._labels[start:end])
			else:
				return self._data[start:end], self._labels[start:end]

	def reset(self):
		self._epochs_completed = 0
		self._index_in_epoch = 0

def retry(initial_delay, max_delay, factor=2.0, jitter=0.25, is_retriable=None):
	"""Simple decorator for wrapping retriable functions.
	Args:
		initial_delay: the initial delay.
		factor: each subsequent retry, the delay is multiplied by this value.
			(must be >= 1).
		jitter: to avoid lockstep, the returned delay is multiplied by a random
			number between (1-jitter) and (1+jitter). To add a 20% jitter, set
			jitter = 0.2. Must be < 1.
		max_delay: the maximum delay allowed (actual max is
			max_delay * (1 + jitter).
		is_retriable: (optional) a function that takes an Exception as an argument
			and returns true if retry should be applied.
	"""
	if factor < 1:
		raise ValueError('factor must be >= 1; was %f' % (factor,))

	if jitter >= 1:
		raise ValueError('jitter must be < 1; was %f' % (jitter,))

	# Generator to compute the individual delays
	def delays():
		delay = initial_delay
		while delay <= max_delay:
			yield delay * random.uniform(1 - jitter,  1 + jitter)
			delay *= factor

	def wrap(fn):
		"""Wrapper function factory invoked by decorator magic."""

		def wrapped_fn(*args, **kwargs):
			"""The actual wrapper function that applies the retry logic."""
			for delay in delays():
				try:
					return fn(*args, **kwargs)
				except Exception as e:  # pylint: disable=broad-except)
					if is_retriable is None:
						continue

					if is_retriable(e):
						time.sleep(delay)
					else:
						raise
			return fn(*args, **kwargs)
		return wrapped_fn
	return wrap

_RETRIABLE_ERRNOS = {
	110,  # Connection timed out [socket.py]
}

def _is_retriable(e):
	return isinstance(e, IOError) and e.errno in _RETRIABLE_ERRNOS

@retry(initial_delay=1.0, max_delay=16.0, is_retriable=_is_retriable)
def urlretrieve_with_retry(url, filename=None):
	return urllib.urlretrieve(url, filename)

def maybe_download(filename, work_directory, source_url):
	"""Download the data from source url, unless it's already here.
	Args:
		filename: string, name of the file in the directory.
		work_directory: string, path to working directory.
		source_url: url to download from if file doesn't exist.
	Returns:
		Path to resulting file.
	"""
	if not gfile.Exists(work_directory):
		gfile.MakeDirs(work_directory)
	filepath = os.path.join(work_directory, filename)
	if not gfile.Exists(filepath):
		print 'Downloading', filename
		temp_file_name, _ = urlretrieve_with_retry(source_url)
		gfile.Copy(temp_file_name, filepath)
		with gfile.GFile(filepath) as f:
			size = f.size()
		print 'Successfully downloaded', filename, size, 'bytes.'
	return filepath

def dense_to_one_hot(labels_dense, num_classes):
	"""Convert class labels from scalars to one-hot vectors."""
	num_labels = labels_dense.shape[0]
	index_offset = np.arange(num_labels) * num_classes
	labels_one_hot = np.zeros((num_labels, num_classes))
	labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
	return labels_one_hot.astype(np.float32)

def extract_cifar_10(train_dir, filename):
	"""Extract the data unless it's already here, and reshape it.
	Args:
		filename: string, name of the file in the directory.
		train_dir: string, path to data directory.
	Returns:
		train_images, train_labels, test_images, test_labels
	"""

	work_directory = train_dir + 'cifar-10-batches-py/'

	if not (os.path.isfile(work_directory + 'data_batch_1') 
		and os.path.isfile(work_directory + 'data_batch_2')
		and os.path.isfile(work_directory + 'data_batch_3')
		and os.path.isfile(work_directory + 'data_batch_4')
		and os.path.isfile(work_directory + 'data_batch_5')
		and os.path.isfile(work_directory + 'test_batch')):
		print 'Extracting', filename
		tar = tarfile.open(train_dir + filename, 'r:gz')
		tar.extractall(path=train_dir)
		tar.close()
		print 'Finished extracting', filename

	train_sets = []

	print 'Unpickling and reshaping data'

	with open(work_directory + 'data_batch_1', 'rb') as f:
		train_sets.append(cPickle.load(f))

	with open(work_directory + 'data_batch_2', 'rb') as f:
		train_sets.append(cPickle.load(f))

	with open(work_directory + 'data_batch_3', 'rb') as f:
		train_sets.append(cPickle.load(f))

	with open(work_directory + 'data_batch_4', 'rb') as f:
		train_sets.append(cPickle.load(f))

	with open(work_directory + 'data_batch_5', 'rb') as f:
		train_sets.append(cPickle.load(f))

	with open(work_directory + 'test_batch', 'rb') as f:
		test_set = cPickle.load(f)

	train_images = np.concatenate([batch['data'].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
								   for batch in train_sets])
	train_labels = dense_to_one_hot(np.concatenate([np.array(batch['labels'])
													for batch in train_sets]), 10)
	test_images = test_set['data'].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
	test_labels = dense_to_one_hot(np.array(test_set['labels']), 10)

	print 'Finished unpickling and reshaping data'

	return train_images, train_labels, test_images, test_labels

Datasets = collections.namedtuple('Datasets', ['train', 'validation', 'test'])

def get_cifar_10(train_dir, augment_image=False, validation_size=0):
	DATASET = 'cifar-10-python.tar.gz'

	local_file = maybe_download(DATASET, train_dir, 'https://www.cs.toronto.edu/~kriz/' + DATASET)
	train_images, train_labels, test_images, test_labels = extract_cifar_10(train_dir, DATASET)

	val_images = train_images[:validation_size]
	val_labels = train_labels[:validation_size]
	train_images = train_images[validation_size:]
	train_labels = train_labels[validation_size:]

	train = DataSet(train_images, train_labels, augment_image)
	val = DataSet(val_images, val_labels, augment_image)
	test = DataSet(test_images, test_labels, augment_image)

	return Datasets(train=train, validation=val, test=test)
