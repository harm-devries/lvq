from fuel.datasets.base import Dataset
import cPickle
import gzip
import numpy


class MNIST(Dataset):

	provides_sources = ('features', 'mask')

	def __init__(self, which_set, **kwargs):
		super(MNIST, self).__init__(self.provides_sources, **kwargs)
		self.which_set = which_set
		self.load()


	def load(self):
		(train_x, train_y), (valid_x, valid_y), (test_x, test_y) = cPickle.load(gzip.open("/home/user/Data/mnist/mnist.pkl.gz", 'rb'))
		if self.which_set == 'train':
			self.data = train_x
			self.mask = numpy.zeros((self.data.shape[0], 10), 'float32')
			for i, y in enumerate(train_y):
				self.mask[i, y] = 1
		elif self.which_set == 'valid':
			self.data = valid_x
			self.mask = numpy.zeros((self.data.shape[0], 10), 'float32')
			for i, y in enumerate(valid_y):
				self.mask[i, y] = 1
		elif self.which_set == 'test':
			self.data = test_x
			self.mask = numpy.zeros((self.data.shape[0], 10), 'float32')
			for i, y in enumerate(test_y):
				self.mask[i, y] = 1
		self.num_examples = self.data.shape[0]

	def get_data(self, state, request):
		return self.filter_sources((self.data[request], self.mask[request]))

