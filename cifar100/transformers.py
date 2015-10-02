import numpy
import scipy
from fuel.transformers import Transformer

class ZCA(object):
    def __init__(self, n_components=None, data=None, filter_bias=0.1):
        self.filter_bias = numpy.float32(filter_bias)
        self.P = None
        self.P_inv = None
        self.n_components = 0
        self.is_fit = False
        if n_components and data:
            self.fit(n_components, data)

    def fit(self, n_components, data):
        if len(data.shape) == 2:
            self.reshape = None
        else:
            assert n_components == numpy.product(data.shape[1:]), \
                'ZCA whitening components should be %d for convolutional data'\
                % numpy.product(data.shape[1:])
            self.reshape = data.shape[1:]

        data = self._flatten_data(data)
        assert len(data.shape) == 2
        n, m = data.shape
        self.mean = numpy.mean(data, axis=0)

        bias = self.filter_bias * scipy.sparse.identity(m, 'float32')
        cov = numpy.cov(data, rowvar=0, bias=1) + bias
        eigs, eigv = scipy.linalg.eigh(cov)

        assert not numpy.isnan(eigs).any()
        assert not numpy.isnan(eigv).any()
        assert eigs.min() > 0

        if self.n_components:
            eigs = eigs[-self.n_components:]
            eigv = eigv[:, -self.n_components:]

        sqrt_eigs = numpy.sqrt(eigs)
        self.P = numpy.dot(eigv * (1.0 / sqrt_eigs), eigv.T)
        assert not numpy.isnan(self.P).any()
        self.P_inv = numpy.dot(eigv * sqrt_eigs, eigv.T)

        self.P = numpy.float32(self.P)
        self.P_inv = numpy.float32(self.P_inv)

        self.is_fit = True

    def apply(self, data, remove_mean=True):
        data = self._flatten_data(data)
        d = data - self.mean if remove_mean else data
        return self._reshape_data(numpy.dot(d, self.P))

    def inv(self, data, add_mean=True):
        d = numpy.dot(self._flatten_data(data), self.P_inv)
        d += self.mean if add_mean else 0.
        return self._reshape_data(d)

    def _flatten_data(self, data):
        if self.reshape is None:
            return data
        assert data.shape[1:] == self.reshape
        return data.reshape(data.shape[0], numpy.product(data.shape[1:]))

    def _reshape_data(self, data):
        assert len(data.shape) == 2
        if self.reshape is None:
            return data
        return numpy.reshape(data, (data.shape[0],) + self.reshape)


class ContrastNorm(object):
    def __init__(self, scale=55, epsilon=1e-8):
        self.scale = numpy.float32(scale)
        self.epsilon = numpy.float32(epsilon)

    def apply(self, data, copy=False):
        if copy:
            data = numpy.copy(data)
        data_shape = data.shape
        if len(data.shape) > 2:
            data = data.reshape(data.shape[0], numpy.product(data.shape[1:]))

        assert len(data.shape) == 2, 'Contrast norm on flattened data'

        data -= data.mean(axis=1)[:, numpy.newaxis]

        norms = numpy.sqrt(numpy.sum(data ** 2, axis=1)) / self.scale
        norms[norms < self.epsilon] = numpy.float32(1.)

        data /= norms[:, numpy.newaxis]

        if data_shape != data.shape:
            data = data.reshape(data_shape)

        return data
        
class Whitening(Transformer):
    """ Makes a copy of the examples in the underlying dataset and whitens it
        if necessary.
    """
    def __init__(self, data_stream, iteration_scheme, whiten, cnorm=None,
                 **kwargs):
        super(Whitening, self).__init__(data_stream,
                                        iteration_scheme=iteration_scheme,
                                        **kwargs)
        data = data_stream.get_data(slice(data_stream.dataset.num_examples))
        self.data = []
        for s, d in zip(self.sources, data):
            if 'features' == s:
                # Fuel provides Cifar in uint8, convert to float32
                d = numpy.require(d, dtype=numpy.float32)
                if cnorm is not None:
                    d = cnorm.apply(d)
                if whiten is not None:
                    d = whiten.apply(d)
                self.data += [d]
            elif 'fine_labels' == s:
                self.data += [d]
            else:
                raise Exception("Unsupported Fuel target: %s" % s)

    def get_data(self, request=None):
        return (s[request] for s in self.data)