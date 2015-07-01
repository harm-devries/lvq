from fuel.datasets.base import Dataset
import cPickle
import gzip
import numpy


class MNIST(Dataset):

	provides_sources = ('features', 'mask')

	def __init__(self, which_set, drop_input=True, n_protos=1, **kwargs):
		super(MNIST, self).__init__(self.provides_sources, **kwargs)
		self.which_set = which_set
		self.n_protos = n_protos
		self.drop_input = drop_input
		self.load()


	def load(self):
		(train_x, train_y), (valid_x, valid_y), (test_x, test_y) = cPickle.load(gzip.open("/home/user/Data/mnist/mnist.pkl.gz", 'rb'))
		if self.which_set == 'train':
			self.data = numpy.vstack((train_x, valid_x))
			self.mask = numpy.zeros((self.data.shape[0], 10*self.n_protos), 'float32')
			train_y = numpy.concatenate((train_y, valid_y))
			for i, y in enumerate(train_y):
				self.mask[i, y:y+self.n_protos] = 1
		elif self.which_set == 'valid':
			self.data = valid_x
			self.mask = numpy.zeros((self.data.shape[0], 10*self.n_protos), 'float32')
			for i, y in enumerate(valid_y):
				self.mask[i, y:y+self.n_protos] = 1
		elif self.which_set == 'test':
			self.data = test_x
			self.mask = numpy.zeros((self.data.shape[0], 10*self.n_protos), 'float32')
			for i, y in enumerate(test_y):
				self.mask[i, y:y+self.n_protos] = 1
		self.num_examples = self.data.shape[0]

	def get_data(self, state, request):
		if self.drop_input == True:
			data = numpy.random.binomial(n=1, p=0.8, size=self.data[request].shape).astype('float32')*self.data[request]/0.8
		else:
			data = self.data[request]
		return self.filter_sources((data, self.mask[request]))

