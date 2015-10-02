######################
# Model 
######################
import theano
from theano.sandbox.rng_mrg import MRG_RandomStreams
import numpy
import theano.tensor as tensor

from lvq.lvq import LVQ, SupervisedNG, initialize_prototypes
from lvq.conv import ConvolutionalSequence, ConvolutionalActivation

from blocks.bricks import MLP, Rectifier, Softmax, Logistic
from blocks.bricks.cost import MisclassificationRate
from blocks.bricks.conv import MaxPooling, Flattener
from blocks.config import config
from blocks.initialization import Uniform, Constant
from blocks.model import Model

rng = MRG_RandomStreams(1)

x = tensor.tensor4('features')
train_x = x #* rng.binomial(x.shape, p=0.8, dtype=theano.config.floatX)
test_x = x#*0.8
y = tensor.imatrix('fine_labels')
flat_y = tensor.flatten(y, outdim=1)

act = Rectifier(name='rect')
layers = [ConvolutionalActivation(act, (3, 3), 128, name='l1'), 
          ConvolutionalActivation(act, (3, 3), 128, name='l2'), 
          ConvolutionalActivation(act, (3, 3), 128, name='l3'), 
          MaxPooling((2, 2), name='p1'),
          ConvolutionalActivation(act, (3, 3), 256, name='l4'),
          ConvolutionalActivation(act, (3, 3), 256, name='l5'),
          ConvolutionalActivation(act, (3, 3), 256, name='l6'),
          MaxPooling((2, 2), name='p2'),
          ConvolutionalActivation(act, (3, 3), 256, name='l7'),
          ConvolutionalActivation(act, (2, 2), 256, name='l8'),
          ConvolutionalActivation(act, (1, 1), 256, name='l9'),
          ]

conv_sequence = ConvolutionalSequence(layers, num_channels=3,
                                      image_size=(32,32),
                                      weights_init=Uniform(0, 0.001),
                                      biases_init=Constant(0.0), 
                                      tied_biases=True,
                                      border_mode='valid',
                                      srng=rng,
                                      dropout_layers=[])
conv_sequence.initialize()

for l in conv_sequence.layers:
    print l.get_dim('output')

flat = Flattener()
train_out = flat.apply(conv_sequence.training(train_x))
test_out = flat.apply(conv_sequence.inference(test_x))

mlp = MLP(activations=[None], dims=[numpy.prod(conv_sequence.get_dim('output'))*128, 100])
mlp.weights_init = Uniform(0.0, 0.001)
mlp.biases_init = Constant(0.0)
mlp.initialize()

#train_out = mlp.apply(train_out)
#test_out = mlp.apply(test_out)

lamb = theano.shared(numpy.float32(1e-3))
lvq = SupervisedNG(100, 256, nonlin=False, gamma=0.5, lamb=lamb, name='lvq')
      
test_loss, test_misclass = lvq.apply(test_out, flat_y, 'test')
loss, misclass = lvq.apply(train_out, flat_y, 'train')

model = Model(loss)

######################
# Data
######################

from cifar100 import CIFAR100
from fuel.schemes import ShuffledScheme, SequentialScheme
from fuel.streams import DataStream

train_dataset = CIFAR100('train')
test_dataset = CIFAR100('test')

batch_size = 100 #Batch size for training
batch_size_mon = 4000 # Batch size for monitoring and batch normalization
n_batches = int(numpy.ceil(float(train_dataset.num_examples)/batch_size_mon))

train_stream_mon = DataStream(train_dataset, iteration_scheme=ShuffledScheme(train_dataset.num_examples, batch_size_mon))
train_stream_bn = DataStream(train_dataset, iteration_scheme=ShuffledScheme(train_dataset.num_examples, batch_size_mon))
train_stream = DataStream(train_dataset, iteration_scheme=ShuffledScheme(train_dataset.num_examples, batch_size))
test_stream = DataStream(test_dataset, iteration_scheme=ShuffledScheme(test_dataset.num_examples, batch_size_mon))

initialize_prototypes(lvq, x, train_out, train_stream_mon, key='fine_labels')

######################
# Training
######################

from blocks.main_loop import MainLoop
from blocks.extensions import FinishAfter,Timing, Printing
from blocks.extensions.monitoring import DataStreamMonitoring, TrainingDataMonitoring
from blocks.algorithms import GradientDescent, Momentum, RMSProp, Adam
from blocks.extensions.saveload import Checkpoint

from lvq.extensions import EarlyStopping, LRDecay, MomentumSwitchOff, NCScheduler
from lvq.batch_norm import BatchNormExtension

lr = theano.shared(numpy.float32(1e-3))
step_rule = Adam(learning_rate=lr)
num_epochs = 350

main_loop = MainLoop(
     model=model,
     data_stream=train_stream,
     algorithm=GradientDescent(
        cost=model.outputs[0], parameters=model.parameters, step_rule=step_rule),
     extensions=[FinishAfter(after_n_epochs=num_epochs),
                 BatchNormExtension(model, train_stream_bn, n_batches),
                 LRDecay(step_rule.learning_rate, decay_epochs=[150, 250]), 
                 #MomentumSwitchOff(step_rule.momentum, 330),
                 NCScheduler(lamb, 5.0, 50, num_epochs),
                 DataStreamMonitoring(
                    variables=[test_loss, test_misclass],
                    data_stream=train_stream_mon,
                    prefix='train'),
                 DataStreamMonitoring(
                    variables=[test_loss, test_misclass],
                    data_stream=test_stream,
                    prefix='test'),
                 Timing(), 
                 EarlyStopping('test_lvq_apply_misclass', 100, './exp/sng_bn_dropout.pkl'),
                 Printing(after_epoch=True)])
main_loop.run()