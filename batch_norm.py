# Code from Cesar Laurent
# 

import logging
import numpy

from blocks.bricks import Activation, Initializable, Feedforward, Linear, Sequence
from blocks.bricks.base import Brick, application, lazy
from blocks.extensions import SimpleExtension
from blocks.extensions.monitoring import MonitoringExtension
from blocks.filter import get_brick
from blocks.graph import ComputationGraph
from blocks.monitoring.evaluators import DatasetEvaluator
from blocks.roles import add_role, WEIGHT, BIAS, PARAMETER
from blocks.utils import dict_subset

from toolz import interleave
from picklable_itertools.extras import equizip

from theano import config, shared, tensor, function

floatX = config.floatX
logger = logging.getLogger()

class MLP(Initializable, Feedforward):
    """Multi-layer perceptron with batch normalization.

    Parameters
    ----------
    activations : list of :class:`.Brick`, :class:`.BoundApplication`,
                  or ``None``
        A list of activations to apply after each linear transformation.
        Give ``None`` to not apply any activation. It is assumed that the
        application method to use is ``apply``. Required for
        :meth:`__init__`.
    dims : list of ints
        A list of input dimensions, as well as the output dimension of the
        last layer. Required for :meth:`~.Brick.allocate`.

    Notes
    -----
    See :class:`Initializable` for initialization parameters.

    Note that the ``weights_init``, ``biases_init`` and ``use_bias``
    configurations will overwrite those of the layers each time the
    :class:`MLP` is re-initialized. For more fine-grained control, push the
    configuration to the child layers manually before initialization.

    >>> from blocks.initialization import IsotropicGaussian, Constant
    >>> mlp = MLP(activations=[Tanh(), None], dims=[30, 20, 10],
    ...           weights_init=IsotropicGaussian(),
    ...           biases_init=Constant(1))
    >>> mlp.push_initialization_config()  # Configure children
    >>> mlp.children[0].weights_init = IsotropicGaussian(0.1)
    >>> mlp.initialize()

    """
    @lazy(allocation=['dims'])
    def __init__(self, activations, dims, **kwargs):
        super(MLP, self).__init__(**kwargs)
        self.activations = activations
        self.linear_transformations = [Linear(name='linear_{}'.format(i))
                                       for i in range(len(activations))]
        self.batch_norms = [BatchNorm(name='bn_{}'.format(i))
                            for i in range(len(activations))]

        self.children.extend([a for a in self.activations if a is not None])
        self.children.extend(self.batch_norms)
        self.children.extend(self.linear_transformations)

        if not dims:
            dims = [None] * (len(activations) + 1)
        self.dims = dims


    @property
    def input_dim(self):
        return self.dims[0]

    @input_dim.setter
    def input_dim(self, value):
        self.dims[0] = value

    @property
    def output_dim(self):
        return self.dims[-1]

    @output_dim.setter
    def output_dim(self, value):
        self.dims[-1] = value
        
    @application(inputs=['input_'], outputs=['output'])    
    def inference(self, input_):
        out = input_
        for brick in interleave([self.linear_transformations, self.batch_norms, self.activations]):
	    if brick is None:
                continue
            if isinstance(brick, BatchNorm):
                out = brick.inference(out)
            else:
                out = brick.apply(out)
        return out
        
    @application(inputs=['input_'], outputs=['output'])    
    def apply(self, input_):
        out = input_
        for brick in interleave([self.linear_transformations, self.batch_norms, self.activations]):
            if brick is None:
                continue
            if isinstance(brick, Brick):
                out = brick.apply(out)
        return out
        
    def _push_allocation_config(self):
        if not len(self.dims) - 1 == len(self.linear_transformations):
            raise ValueError
        for input_dim, output_dim, layer in \
                equizip(self.dims[:-1], self.dims[1:],
                        self.linear_transformations):
            layer.input_dim = input_dim
            layer.output_dim = output_dim
            layer.use_bias = self.use_bias
        for dim, bn in equizip(self.dims[1:], self.batch_norms):
            bn.input_dim = dim


class BatchNorm(Activation):
    """Brick for Batch Normalization. It works with 4D Tensors (conv.) and
    2D Tensors (fully connected layers).
    The Batch Normalization paper:
    S. Ioffe, C. Szegedy, Batch Normalization: Accelerating Deep Network
    Training by Reducing Internal Covariate Shift.
    Parameters
    ----------
    input_dim : int
        The number of features (or features maps for convolutions).
    n_batches : int
        The number of batches used to update the pop. means and vars.
    epsilon : float
        Small constant for sqrt stability.
    Examples
    --------
    >>> import theano
    >>> from theano import tensor
    >>> x = tensor.vector('x')
    Creating a network:
    >>> y = Linear(input_dim=10, output_dim=5).apply(x)
    >>> bn = BatchNorm(input_dim=5)
    >>> train_out = bn.apply(y)
    Creating both train and test computation graphs:
    >>> train_cg = ComputationGraph([train_out])
    >>> test_cg = create_inference_graph(train_cg, [bn])
    Preparing the update extension:
    >>> batch_size = 50 #The size of the batches
    >>> n_batches = 10 #The number of batches to use to update the stats.
    >>> scheme = ShuffledScheme(batch_size*n_batches, batch_size)
    >>> stream = DataStream(DATASET, iteration_scheme=scheme)
    >>> extensions.insert(0, BatchNormExtension([bn], stream, n_batches))
    """
    @lazy(allocation=['input_dim'])
    def __init__(self, input_dim, epsilon=1e-6, **kwargs):
        super(BatchNorm, self).__init__(**kwargs)
        self.input_dim = input_dim
        self.epsilon = epsilon

    @property
    def gamma(self):
        return self.parameters[0]

    @property
    def beta(self):
        return self.parameters[1]

    def _allocate(self):
        gamma_val = numpy.ones(self.input_dim, dtype=floatX)
        gamma = shared(name='gamma', value=gamma_val)
        beta_val = numpy.zeros(self.input_dim, dtype=floatX)
        beta = shared(name='beta', value=beta_val)
        add_role(gamma, PARAMETER)
        add_role(beta, PARAMETER)
        self.parameters.append(gamma)
        self.parameters.append(beta)
        # Keeping track of the means and variances during the training.
        means_val = numpy.zeros(self.input_dim, dtype=floatX)
        self.pop_means = shared(name='means', value=means_val)
        vars_val = numpy.ones(self.input_dim, dtype=floatX)
        self.pop_vars = shared(name='variances', value=vars_val)

    def get_updates(self, n_batches):
        """Update the population means and variances of the brick. Use
        n_batches from the training dataset to do so.
        """
        m_u = (self.pop_means, (self.pop_means
                                + 1./n_batches * self.batch_means))
        v_u = (self.pop_vars, (self.pop_vars
                               + 1./n_batches * self.batch_vars))
        return [m_u, v_u]

    def _inference(self, input_):
        output = (input_ - self.pop_means.dimshuffle(*self.pattern))
        output /= tensor.sqrt(self.pop_vars.dimshuffle(*self.pattern)
                              + self.epsilon)
        output *= self.gamma.dimshuffle(*self.pattern)
        output += self.beta.dimshuffle(*self.pattern)
        return output

    def _training(self, input_):
        self.batch_means = input_.mean(axis=self.axes, keepdims=False,
                                       dtype=floatX)
        self.batch_vars = input_.var(axis=self.axes, keepdims=False)
        output = input_ - self.batch_means.dimshuffle(*self.pattern)
        output /= tensor.sqrt(self.batch_vars.dimshuffle(*self.pattern)
                              + self.epsilon)
        output *= self.gamma.dimshuffle(*self.pattern)
        output += self.beta.dimshuffle(*self.pattern)
        return output

    def _check_input(self, x):
        if x.ndim == 2:
            self.axes = [0]
            self.pattern = ['x', 0]
        elif x.ndim == 4:
            self.axes = [0, 2, 3]
            self.pattern = ['x', 0, 'x', 'x']
        elif x.ndim == 3:
            self.axes = [0, 1]
            self.pattern = ['x', 'x', 0]
        else:
            raise NotImplementedError

    #@application(inputs=['input_'], outputs=['output'])
    def apply(self, input_):
        self._check_input(input_)
        self.training_output = self._training(input_)
        return self.training_output
        
    #@application(inputs=['input_'], outputs=['output'])
    def inference(self, input_):
        self._check_input(input_)
        return self._inference(input_)
       


class BatchNormExtension(SimpleExtension, MonitoringExtension):
    """Computes the population means and variance of the BatchNorm bricks
    in the network. This extension must be placed before any other
    monitoring.
    
    Parameters
    ----------
    graph : instance of :class:`ComputationGraph`
        The training computation graph.
    data_stream : instance of :class:`DataStream`
        The data stream used to compute the population statistics on. It
        should provide n_batches only.
    n_batches: int
        The number of batches used to update the population statistics.
    """
    def __init__(self, graph, data_stream, n_batches, **kwargs):
        kwargs.setdefault("after_epoch", True)
        kwargs.setdefault("before_first_epoch", True)
        super(BatchNormExtension, self).__init__(**kwargs)
        self.n_batches = n_batches
        self.bricks = get_batch_norm_bricks(graph)
        self.data_stream = data_stream
        self.updates = self._get_updates()
        variables = [brick.training_output for brick in self.bricks]
        self._computation_graph = ComputationGraph(variables)
        self.inputs = self._computation_graph.inputs
        self.inputs = list(set(self.inputs))
        self.inputs_names = [v.name for v in self.inputs]
        self._compile()

    def _get_updates(self):
        updates = []
        for brick in self.bricks:
            updates.extend(brick.get_updates(self.n_batches))
        return updates

    def _reset(self, x):
        x.set_value(numpy.zeros(x.get_value().shape, dtype=floatX))

    def _compile(self):
        self._fun = function(self.inputs, [], updates=self.updates,
                             on_unused_input='ignore')

    def _evaluate(self):
        for batch in self.data_stream.get_epoch_iterator(as_dict=True):
            batch = dict_subset(batch, self.inputs_names)
            self._fun(**batch)

    def do(self, which_callback, *args):
        logger.info('Computation of population statistics started')
        # 1. Reset the pop means and vars
        for brick in self.bricks:
            self._reset(brick.pop_means)
            self._reset(brick.pop_vars)
        # 2. Update them
        self._evaluate()
        logger.info('Computation of population statistics finished')


def create_inference_graph(graph):
    """Create the inference graph from the training computation graph.
    Parameters
    ----------
    graph : instance of :class:`ComputationGraph`
        The training computation graph.
    """
    replacements = {}
    bricks = get_batch_norm_bricks(graph)
    for brick in bricks:
        replacements.update(brick.get_replacements())
    return graph.replace(replacements)


def get_batch_norm_bricks(graph):
    """Returns the batch norm bricks (BatchNorm and BatchNorm3D) in a
       computation graph.
    Parameters
    ----------
    graph : instance of :class:`ComputationGraph`
        The training computation graph.
    """
    bricks = []
    for variable in graph.variables:
        brick = get_brick(variable)
        if isinstance(brick, BatchNorm):
            if brick not in bricks:
                bricks.append(brick)
    return bricks
