import matplotlib.pyplot as plt
import numpy as np
import cPickle

def get_labels():
	with open('data/cifar_10/cifar-10-batches-py/batches.meta', 'rb') as f:
		label_names = cPickle.load(f)

	return label_names['label_names']

def visualize_cifar_10(images, pred_labels, true_labels):
	assert images.shape[0] == pred_labels.shape[0] and images.shape[0] == true_labels.shape[0], 'Images/labels dimension mismatch'

	label_names = get_labels()

	for i in range(images.shape[0]):
		plt.imshow(images[i], interpolation='nearest')
		title = 'pred: ' + label_names[np.argmax(pred_labels[i])] + ', true: ' + label_names[np.argmax(true_labels[i])]
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