from fuel.datasets.base import Dataset
from fuel.utils import do_not_pickle_attributes
import cPickle
import gzip
import numpy

@do_not_pickle_attributes('data', 'targets')
class CIFAR100(Dataset):
    provides_sources = ('features', 'fine_labels')

    def __init__(self, which_set, **kwargs):
        if 'sources' in kwargs:
            super(CIFAR100, self).__init__(**kwargs)
        else:
            super(CIFAR100, self).__init__(self.provides_sources, **kwargs)
        self.which_set = which_set
        self.load()


    def load(self):
        #(train_x, train_y), (valid_x, valid_y), (test_x, test_y) = cPickle.load(gzip.open("/home/user/Data/mnist/mnist.pkl.gz", 'rb'))
        d = cPickle.load(open("/Tmp/devries/cifar100.pkl", 'rb')) 
        if self.which_set == 'train':
                self.data = d['train_x']
                self.targets = d['train_y'].astype('int32')
        elif self.which_set == 'test':
                self.data = d['test_x']
                self.targets = d['test_y'].astype('int32')
        self.num_examples = self.data.shape[0]

    def get_data(self, state, request):
        return self.filter_sources((self.data[request, :], self.targets[request]))

