######################
# Model 
######################
import theano
import theano.tensor as tensor
from theano.sandbox.rng_mrg import MRG_RandomStreams
from blocks.bricks import Rectifier, Softmax, Linear
from blocks.bricks.cost import MisclassificationRate
from blocks.config import config
from blocks.initialization import Uniform, Constant
from blocks.model import Model
from blocks.graph import Annotation, add_annotation

from lvq.initialization import NormalizedGaussian
from lvq.batch_norm import MLP
#from blocks.bricks import MLP

seed = config.default_seed
rng = MRG_RandomStreams(seed)

x = tensor.tensor4('features')
flat_x = tensor.flatten(x, outdim=2)
flat_x_noise = flat_x + rng.normal(size=flat_x.shape, std=0.5)
y = tensor.imatrix('targets')
flat_y = tensor.flatten(y, outdim=1)

rect = Rectifier()
mlp = MLP(dims=[784, 1200, 1200, 200], activations=[rect, rect, rect], seed=10)
mlp.weights_init = Uniform(0.0, 0.01)
mlp.biases_init = Constant(0.0)
mlp.initialize()

lin = Linear(200, 10, use_bias=True)
lin.weights_init = Uniform(0.0, 0.01)
lin.biases_init = Constant(0.0)
lin.initialize()

train_out = lin.apply(mlp.apply(flat_x))
test_out = lin.apply(mlp.apply(flat_x))

sm = Softmax(name='softmax')
loss = sm.categorical_cross_entropy(flat_y, train_out).mean()
loss.name = 'nll'
misclass = MisclassificationRate().apply(flat_y, train_out)
misclass.name = 'misclass'

test_loss = sm.categorical_cross_entropy(flat_y, test_out).mean()
test_loss.name = 'nll'
test_misclass = MisclassificationRate().apply(flat_y, test_out)
test_misclass.name = 'misclass'

model = Model(loss)

######################
# Data
######################
import numpy 
#from mnist import MNIST
from fuel.datasets.mnist import MNIST
from fuel.transformers import ScaleAndShift, ForceFloatX
from fuel.streams import DataStream
from fuel.schemes import ShuffledScheme

mnist_train = MNIST(['train'])
#mnist_valid = MNIST('valid', n_protos=1, drop_input=False)
mnist_test = MNIST(['test'])

batch_size = 100 #Batch size for training
batch_size_mon = 2000 # Batch size for monitoring and batch normalization
n_batches = int(numpy.ceil(float(mnist_train.num_examples)/batch_size_mon))

ind = range(mnist_train.num_examples)
train_ind = ind[:50000]
val_ind = ind[50000:]
    
def preprocessing(data_stream):
    return ForceFloatX(ScaleAndShift(data_stream, 1/255.0, 0.0, which_sources=('features',)), which_sources=('features',))

train_stream_mon = preprocessing(DataStream(mnist_train, iteration_scheme=ShuffledScheme(train_ind, batch_size_mon)))
train_stream_bn = preprocessing(DataStream(mnist_train, iteration_scheme=ShuffledScheme(train_ind, batch_size_mon)))
train_stream = preprocessing(DataStream(mnist_train, iteration_scheme=ShuffledScheme(train_ind, batch_size)))
valid_stream =  preprocessing(DataStream(mnist_train, iteration_scheme=ShuffledScheme(val_ind, batch_size)))
test_stream = preprocessing(DataStream(mnist_test, iteration_scheme=ShuffledScheme(mnist_test.num_examples, batch_size_mon)))

######################
# Training
######################

from blocks.main_loop import MainLoop
from blocks.extensions import FinishAfter,Timing, Printing
from blocks.extensions.monitoring import DataStreamMonitoring
from blocks.algorithms import GradientDescent, Momentum, Adam
from blocks.extensions.saveload import Checkpoint

from lvq.batch_norm import BatchNormExtension
from lvq.extensions import EarlyStopping, LRDecay, MomentumSwitchOff
learning_rate = theano.shared(numpy.float32(1e-4))
step_rule = Momentum(5e-2, 0.9)

main_loop = MainLoop(
     model=model,
     data_stream=train_stream,
     algorithm=GradientDescent(
        cost=model.outputs[0], parameters=model.parameters, step_rule=step_rule),
     extensions=[FinishAfter(after_n_epochs=100),
                 BatchNormExtension(model, train_stream_bn, n_batches),
                 LRDecay(step_rule.learning_rate, [20, 40, 60, 80]), 
                 #MomentumSwitchOff(step_rule.momentum, 140),
                 DataStreamMonitoring(
                    variables=[test_loss, test_misclass],
                    data_stream=train_stream_mon,
                    prefix='train'),
                 DataStreamMonitoring(
                    variables=[test_loss, test_misclass],
                    data_stream=valid_stream,
                    prefix='valid'),
                 DataStreamMonitoring(
                    variables=[test_loss, test_misclass],
                    data_stream=test_stream,
                    prefix='test'),
                 EarlyStopping('valid_misclass', 100, './exp/softmax_2_bn_noise.pkl'),
                 Timing(), 
                 Printing(after_epoch=True)])
main_loop.run()
