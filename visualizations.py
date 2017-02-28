import matplotlib.pyplot as plt
import numpy as np
import cPickle

def get_labels(dataset):
	'''
	Args:
		dataset: String. One of 'mnist', 'cifar_10'
	'''

	if dataset == 'mnist':
		return {i: str(i) for i in range(10)}
	elif dataset == 'cifar_10':
		with open('data/cifar_10/cifar-10-batches-py/batches.meta', 'rb') as f:
			label_names = cPickle.load(f)
		return label_names['label_names']
	else:
		raise ValueError('Dataset not found')

def show_images(dataset, images, true_labels, pred_labels=None):

	if pred_labels is None:
		assert images.shape[0] == true_labels.shape[0], 'Images/labels dimension mismatch'
	else:
		assert images.shape[0] == true_labels.shape[0] and images.shape[0] == pred_labels.shape[0], 'Images/labels dimension mismatch'

	label_names = get_labels(dataset)

	if dataset == 'mnist':
		images = images.reshape([-1, 28, 28])

	for i in range(images.shape[0]):
		if dataset == 'mnist':
			plt.imshow(images[i], interpolation='nearest', cmap='gray')
		else:
			plt.imshow(images[i], interpolation='nearest')
		if pred_labels is None:
			title = 'true: ' + label_names[np.argmax(true_labels[i])]
		else:
			title = 'true: ' + label_names[np.argmax(true_labels[i])] + ', pred: ' + label_names[np.argmax(pred_labels[i])]
		plt.title(title)
		plt.show()

def plot_learning_curve(curves, x_axis=None, title='Loss'):
	if x_axis is None:
		for curve in curves:
			plt.plot(curve)
	else:
		for curve in curves:
			plt.plot(x_axis, curve)
	plt.title(title)
	plt.xlabel('Batch #')
	plt.show()