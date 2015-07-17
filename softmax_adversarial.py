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


x = tensor.fmatrix('features')
#x = x.reshape((x.shape[0], 784))
#mask = tensor.fmatrix('mask')
y = tensor.imatrix('targets')
y_flat = y.flatten()

l1 = LinearMaxout(784, 300, 5, name='l1')
l1.weights_init = Uniform(0, 0.005)
l1.biases_init = Constant(0.0)
l1.initialize()


l3 = Linear(300, 200, name='l3')
l3.weights_init = Uniform(0, 0.005)
l3.biases_init = Constant(0.0)
l3.initialize()

out = l3.apply(l1.apply(x))

# mlp = MLP(dims=[784, 5000, 100], activations=[Rectifier().apply, None])
# mlp.weights_init = Uniform(0, 0.01)
# mlp.biases_init = Constant(0.0)
# mlp.initialize()

# out = mlp.apply(x)

#lvq = LVQ(10, 200, n_protos=1)
#lvq.weights_init = Uniform(0, 0.1)
#lvq.initialize()
#loss, misclass = lvq.apply(out, mask)

#loss.name = 'loss'
#misclass.name = 'misclass'

sm = Softmax()
loss = sm.categorical_cross_entropy(y_flat, out)
pred = sm.apply(out)
loss.name = 'nll'
misclass = MisclassificationRate().apply(y_flat, out)
misclass.name = 'misclass'

model = Model(loss)

######################
# Data
######################
import numpy 
#from mnist import MNIST
from fuel.datasets.mnist import MNIST
from fuel.streams import DataStream
from fuel.schemes import ShuffledScheme
from fuel.transformers import Mapping, Rename

mnist_train = MNIST('train', n_protos=1, drop_input=False)
mnist_valid = MNIST('valid', n_protos=1, drop_input=False)
mnist_test = MNIST('test', n_protos=1, drop_input=False)


batch_size = 250

train_stream = DataStream(mnist_train, iteration_scheme=ShuffledScheme(mnist_train.num_examples, batch_size))
#train_stream = Rename(train_stream, {'features': 'features', 'targets': 'targets'})
valid_stream = DataStream(mnist_valid, iteration_scheme=ShuffledScheme(mnist_valid.num_examples, batch_size))
test_stream = DataStream(mnist_test, iteration_scheme=ShuffledScheme(mnist_valid.num_examples, batch_size))

######################
# Training
######################


from blocks.main_loop import MainLoop
from blocks.extensions import FinishAfter,Timing, Printing
from blocks.extensions.monitoring import DataStreamMonitoring
from blocks.algorithms import GradientDescent, Momentum, RMSProp

main_loop = MainLoop(
     model=model,
     data_stream=train_stream,
     algorithm=GradientDescent(
        cost=loss, params=model.parameters, step_rule=Momentum(0.01, 0.9)),
     extensions=[FinishAfter(after_n_epochs=100),
                 DataStreamMonitoring(
                    variables=[loss, misclass],
                    data_stream=train_stream,
                    prefix='train'),
                 DataStreamMonitoring(
                    variables=[loss, misclass],
                    data_stream=valid_stream,
                    prefix='valid'),
                 DataStreamMonitoring(
                    variables=[loss, misclass],
                    data_stream=test_stream,
                    prefix='test'),
                 Timing(), 
                 Printing(after_epoch=True)])
main_loop.run()

grad = tensor.grad(loss, [x])

f_grad = theano.function([x, y], grad)
f_pred = theano.function([x], pred)

epsilon = 0.001
num_adv = 0

for x in train_stream.get_epoch_iterator():
    pred = numpy.argmax(f_pred(x[0]), axis=1)
    g = f_grad(x[0], x[1])
    new_x = x[0] + epsilon*g
    pred2 = numpy.argmax(f_pred(new_x), axis=1)
    num_adv += (pred == pred2).sum()
    
print num_adv
    
    
    
    




