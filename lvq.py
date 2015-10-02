import theano
import numpy 

from blocks.bricks import Initializable
from blocks.utils import shared_floatx_nans
from blocks.bricks.base import application

import theano.tensor as tensor

class LVQ(Initializable):

    def __init__(self, n_classes, dim, nonlin=True, gamma=2.0, **kwargs):
        super(LVQ, self).__init__(**kwargs)
        self.n_classes = n_classes
        self.dim = dim
        self.gamma = gamma
        self.nonlin=nonlin

    def _allocate(self):
        W = shared_floatx_nans((self.n_classes, self.dim), name='prototypes')
        self.parameters.append(W)

    def _initialize(self):
        W, = self.parameters
        self.weights_init.initialize(W, self.rng)
        
    @application(inputs=['x', 'y', 'prefix'], outputs=['cost', 'misclass'])
    def apply(self, x, y, prefix):
        W, = self.parameters
        #Create mask from labels
        mask = tensor.alloc(0, y.shape[0]*self.n_classes)
        ind = tensor.arange(y.shape[0])*self.n_classes + y
        mask = tensor.set_subtensor(mask[ind], 1.0)
        mask = tensor.reshape(mask, (y.shape[0], self.n_classes))
        
        #Compute distance matrix
        D = ((W**2).sum(axis=1, keepdims=True).T + (x**2).sum(axis=1, keepdims=True) - 2*tensor.dot(x, W.T))
        self.add_auxiliary_variable(D, name=prefix+'_D')
        d_correct = (D + (1-mask)*numpy.float32(2e30)).min(axis=1)
        d_incorrect = (D + mask*numpy.float32(2e30)).min(axis=1)
        
        c = (d_correct - d_incorrect)
        self.add_auxiliary_variable(c, name=prefix+'_cost')
        if self.nonlin:
            c = tensor.exp(self.gamma*c)
        cost = c.mean()
        misclass = (tensor.switch(d_correct - d_incorrect < 0, 0.0, 1.0)).mean()
        return cost, misclass

class SupervisedNG(Initializable):

    def __init__(self, n_classes, dim, nonlin=True, gamma=2.0, lamb=1.0, **kwargs):
        super(SupervisedNG, self).__init__(**kwargs)
        self.n_classes = n_classes
        self.dim = dim
        self.gamma = gamma
        self.nonlin=nonlin
        self.lamb = lamb
        self.weighting = tensor.exp(-lamb*numpy.arange(n_classes-1)) / tensor.exp(-lamb*numpy.arange(n_classes-1)).sum()

    def _allocate(self):
        W = shared_floatx_nans((self.n_classes, self.dim), name='prototypes')
        self.parameters.append(W)

    def _initialize(self):
        W, = self.parameters
        self.weights_init.initialize(W, self.rng)
        
    @application(inputs=['x', 'y', 'prefix'], outputs=['cost', 'misclass'])
    def apply(self, x, y, prefix):
        W, = self.parameters
        
        mask = tensor.alloc(0, y.shape[0]*self.n_classes)
        ind = tensor.arange(y.shape[0])*self.n_classes + y
        mask = tensor.set_subtensor(mask[ind], 1.0)
        mask = tensor.reshape(mask, (y.shape[0], self.n_classes))
        
        #Compute distance matrix
        D = ((W**2).sum(axis=1, keepdims=True).T + (x**2).sum(axis=1, keepdims=True) - 2*tensor.dot(x, W.T))
        self.add_auxiliary_variable(D, name=prefix+'_D')
              
        d_correct = tensor.reshape(D[mask.nonzero()], (y.shape[0], 1))
        d_incorrect = tensor.reshape(D[(1.0-mask).nonzero()], (y.shape[0], self.n_classes-1))
        
        c = (d_correct - d_incorrect)/(d_correct + d_incorrect)
        c_sorted = tensor.sort(c, axis=1)[:, ::-1]
        
        c = (self.weighting*c_sorted).sum(axis=1, keepdims=True)
        self.add_auxiliary_variable(c, name=prefix+'_cost')
        if self.nonlin:
            c = tensor.exp(self.gamma*c)
        cost = c.mean()
        misclass = (tensor.switch(c_sorted[:, 0] < 0, 0.0, 1.0)).mean()
        return cost, misclass
        
"""
Initialize prototypes as class conditional means
"""
def initialize_prototypes(brick, x, embedding, data_stream, key='targets'):
    protos = numpy.zeros((brick.n_classes, brick.dim)).astype('float32')
    n_examples = numpy.zeros((brick.n_classes,1))
    f_emb = theano.function([x], embedding)
    for data in data_stream.get_epoch_iterator(as_dict=True):
        emb = f_emb(data['features'])
        for h, y in zip(emb, data[key]):
            protos[y, :] += h
            n_examples[y, :] += 1
    brick.parameters[0].set_value((protos/n_examples).astype('float32')) 
    
