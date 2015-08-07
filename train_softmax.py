######################
# Model 
######################

import theano.tensor as tensor
from lvq import LVQ
from blocks.bricks import MLP, Linear, LinearMaxout
from blocks.bricks import Rectifier, Softmax
from blocks.bricks.cost import MisclassificationRate
from blocks.initialization import Uniform, Constant
from blocks.model import Model
from blocks.graph import Annotation, add_annotation


x = tensor.fmatrix('features')
#x_flat = x.reshape((x.shape[0], 784))
y = tensor.ivector('targets')

mlp = MLP(dims=[784, 1000, 10], activations=[Rectifier().apply, None])
mlp.weights_init = Uniform(0, 0.005)
mlp.biases_init = Constant(0.0)
mlp.initialize()
#l1 = LinearMaxout(784, 300, 5, name='l1')
#l1.weights_init = Uniform(0, 0.005)
#l1.biases_init = Constant(0.0)
#l1.initialize()

#l2 = LinearMaxout(300, 300, 5, name='l2')
#l2.weights_init = Uniform(0, 0.005)
#l2.biases_init = Constant(0.0)
#l2.initialize()

#l3 = Linear(784, 10, name='l3')
#l3.weights_init = Uniform(0, 0.005)
#l3.biases_init = Constant(0.0)
#l3.initialize()

out = mlp.apply(x)

sm = Softmax(name='softmax')
pred = sm.apply(out).max(axis=1).mean()
pred.name= 'conf'
loss = sm.categorical_cross_entropy(y, out)
loss.name = 'nll'
misclass = MisclassificationRate().apply(y, out)
misclass.name = 'misclass'

model = Model(loss)

######################
# Data
######################
import numpy 
from mnist import MNIST
#from fuel.datasets.mnist import MNIST
from fuel.streams import DataStream
from fuel.schemes import ShuffledScheme

mnist_train = MNIST('train', n_protos=1, drop_input=False, sources=('features', 'targets'))
#mnist_valid = MNIST('valid', n_protos=1, drop_input=False)
mnist_test = MNIST('test', n_protos=1, drop_input=False, sources=('features', 'targets'))

batch_size = 250

train_stream = DataStream(mnist_train, iteration_scheme=ShuffledScheme(mnist_train.num_examples, batch_size))
#train_stream = Rename(train_stream, {'features': 'features', 'targets': 'targets'})
#valid_stream = DataStream(mnist_valid, iteration_scheme=ShuffledScheme(mnist_valid.num_examples, batch_size))
test_stream = DataStream(mnist_test, iteration_scheme=ShuffledScheme(mnist_test.num_examples, batch_size))

######################
# Training
######################


from blocks.main_loop import MainLoop
from blocks.extensions import FinishAfter,Timing, Printing
from blocks.extensions.monitoring import DataStreamMonitoring
from blocks.algorithms import GradientDescent, Momentum, RMSProp
from blocks.extensions.saveload import Checkpoint

main_loop = MainLoop(
     model=model,
     data_stream=train_stream,
     algorithm=GradientDescent(
        cost=loss, parameters=model.parameters, step_rule=RMSProp(learning_rate=1e-3, max_scaling=1e4)),
     extensions=[FinishAfter(after_n_epochs=50),
                 DataStreamMonitoring(
                    variables=[loss, misclass, pred],
                    data_stream=train_stream,
                    prefix='train'),
                 DataStreamMonitoring(
                    variables=[loss, misclass, pred],
                    data_stream=test_stream,
                    prefix='test'),
                 Checkpoint('./exp/deep_softmax.pkl'),
                 Timing(), 
                 Printing(after_epoch=True)])
main_loop.run()
