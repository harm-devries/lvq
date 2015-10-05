######################
# Model 
######################
import theano
from theano.sandbox.rng_mrg import MRG_RandomStreams
import numpy
import theano.tensor as tensor

from lvq.lvq import LVQ
from lvq.conv import ConvolutionalSequence, ConvolutionalActivation

from blocks.bricks import MLP, Rectifier, Softmax, Logistic
from blocks.bricks.cost import MisclassificationRate
from blocks.bricks.conv import MaxPooling, Flattener
from blocks.config import config
from blocks.initialization import Uniform, Constant
from blocks.model import Model

rng = MRG_RandomStreams(1)

x = tensor.tensor4('features') 
train_x = x*rng.binomial(x.shape, p=0.8, dtype=theano.config.floatX)
test_x = x*0.8
y = tensor.imatrix('fine_labels')
flat_y = tensor.flatten(y, outdim=1)

act = Rectifier(name='rect')
layers = [ConvolutionalActivation(act, (3, 3), 96, name='l1'), 
          ConvolutionalActivation(act, (3, 3), 96, name='l2'), 
          ConvolutionalActivation(act, (3, 3), 96, name='l3'), 
          MaxPooling((2, 2), name='p1'),
          ConvolutionalActivation(act, (3, 3), 192, name='l4'),
          ConvolutionalActivation(act, (3, 3), 192, name='l5'),
          ConvolutionalActivation(act, (3, 3), 192, name='l6'),
          MaxPooling((2, 2), name='p2'),
          ConvolutionalActivation(act, (3, 3), 192, name='l7'),
          ConvolutionalActivation(act, (2, 2), 192, name='l8'),
          ]

conv_sequence = ConvolutionalSequence(layers, num_channels=3,
                                      image_size=(32,32),
                                      weights_init=Uniform(0, 0.001),
                                      biases_init=Constant(0.0), 
                                      tied_biases=True,
                                      border_mode='valid',
                                      srng=rng,
                                      dropout_layers=['p1', 'p2'])
conv_sequence.initialize()

for l in conv_sequence.layers:
    print l.get_dim('output')

flat = Flattener()
train_out = flat.apply(conv_sequence.training(train_x))
test_out = flat.apply(conv_sequence.inference(test_x))

mlp = MLP(activations=[None], dims=[192, 100])
mlp.weights_init = Uniform(0.0, 0.001)
mlp.biases_init = Constant(0.0)
mlp.initialize()

train_out = mlp.apply(train_out)
test_out = mlp.apply(test_out)

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

#from mnist import MNIST
from cifar100 import CIFAR100
from fuel.schemes import ShuffledScheme, SequentialScheme
from fuel.streams import DataStream

train_dataset = CIFAR100('train')
test_dataset = CIFAR100('test')

batch_size = 100 
#Batch size for training
batch_size_mon = 4000 # Batch size for monitoring and batch normalization
n_batches = int(numpy.ceil(float(train_dataset.num_examples)/batch_size_mon))

train_stream_mon = DataStream(train_dataset, iteration_scheme=ShuffledScheme(train_dataset.num_examples, batch_size_mon))
train_stream_bn = DataStream(train_dataset, iteration_scheme=ShuffledScheme(train_dataset.num_examples, batch_size_mon))
train_stream = DataStream(train_dataset, iteration_scheme=ShuffledScheme(train_dataset.num_examples, batch_size))
test_stream = DataStream(test_dataset, iteration_scheme=ShuffledScheme(test_dataset.num_examples, batch_size_mon))

######################
# Training
######################

from blocks.main_loop import MainLoop
from blocks.extensions import FinishAfter,Timing, Printing
from blocks.extensions.monitoring import DataStreamMonitoring, TrainingDataMonitoring
from blocks.algorithms import GradientDescent, Momentum, RMSProp, Adam
from blocks.extensions.saveload import Checkpoint

from lvq.extensions import EarlyStopping, LRDecay, MomentumSwitchOff
from lvq.batch_norm import BatchNormExtension

lr = theano.shared(numpy.float32(1e-3))
step_rule = Momentum(1e-1, 0.9) #Adam(learning_rate=lr) #RMSProp(learning_rate=1e-5, max_scaling=1e4)
num_epochs = 350

main_loop = MainLoop(
     model=model,
     data_stream=train_stream,
     algorithm=GradientDescent(
        cost=model.outputs[0], parameters=model.parameters, step_rule=step_rule),
     extensions=[FinishAfter(after_n_epochs=150),
                 BatchNormExtension(model, train_stream_bn, n_batches),
                 LRDecay(step_rule.learning_rate, [200, 300]), 
                 MomentumSwitchOff(step_rule.momentum, 140),
                 DataStreamMonitoring(
                    variables=[test_loss, test_misclass],
                    data_stream=train_stream_mon,
                    prefix='train'),
                 DataStreamMonitoring(
                    variables=[test_loss, test_misclass],
                    data_stream=test_stream,
                    prefix='test'),
                 Timing(), 
                 EarlyStopping('test_misclass', 100, './exp/softmax_cifar100.pkl'),
                 Printing(after_epoch=True)])
main_loop.run()
