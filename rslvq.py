import numpy 

from blocks.bricks import Initializable
from blocks.utils import shared_floatx_nans, shared_floatx
from blocks.bricks.base import application

import theano
import theano.tensor as tensor

class RSLVQ(Initializable):

    def __init__(self, n_classes, dim, sigma, **kwargs):
        super(RSLVQ, self).__init__(**kwargs)
        self.n_classes = n_classes
        self.dim = dim
        self.sigma = sigma

    def _allocate(self):
        W = shared_floatx_nans((self.n_classes, self.dim), name='prototypes')
        self.parameters.append(W)
       	#sigma = shared_floatx(1.0, name='sigma')
        #self.parameters.append(sigma)

    def _initialize(self):
        W, = self.parameters
        self.weights_init.initialize(W, self.rng)

    @application(inputs=['x'], outputs=['log_prob', 'loss'])
    def apply(self, x, y):
        W, = self.parameters
        D = ((W**2).sum(axis=1, keepdims=True).T + (x**2).sum(axis=1, keepdims=True) - 2*tensor.dot(x, W.T))
        D = -D/(2.0*self.sigma**2)
        D = D - D.max(axis=1, keepdims=True)
        
        log_prob = D - tensor.log(tensor.exp(D).sum(axis=1, keepdims=True))
        flat_log_prob = log_prob.flatten()
        range_ = tensor.arange(y.shape[0])
        flat_indices = y + range_ * D.shape[1]
        cost = -tensor.mean(flat_log_prob[flat_indices])
        
	a = D.shape
        return D, cost
