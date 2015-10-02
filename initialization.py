import theano
import numpy
from blocks.initialization import NdarrayInitialization

class NormalizedGaussian(NdarrayInitialization):
    
    def generate(self, rng, shape):
        print shape
	m = rng.normal(0.0, 1.0/numpy.sqrt(shape[0]), size=shape)
        return m.astype(theano.config.floatX)
