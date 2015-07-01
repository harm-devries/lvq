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
x = x.reshape((x.shape[0], 784))
mask = tensor.fmatrix('mask')
y = tensor.imatrix('targets')
y = y.flatten()

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

lvq = LVQ(10, 200, n_protos=1)
lvq.weights_init = Uniform(0, 0.1)
lvq.initialize()
loss, misclass = lvq.apply(out, mask)

#loss.name = 'loss'
#misclass.name = 'misclass'

#loss = Softmax().categorical_cross_entropy(y, out)
#loss.name = 'nll'
#misclass = MisclassificationRate().apply(y, out)
#misclass.name = 'misclass'

model = Model(loss)

######################
# Data
######################
import numpy 
from mnist import MNIST
#from fuel.datasets.mnist import MNIST
from fuel.streams import DataStream
from fuel.schemes import ShuffledScheme
from fuel.transformers import Mapping, Rename

mnist_train = MNIST('train', n_protos=1, drop_input=False)
mnist_valid = MNIST('valid', n_protos=1, drop_input=False)
mnist_test = MNIST('test', n_protos=1, drop_input=False)

batch_size = 500
num_protos=10

# def mask(x):
#     y= x[1]
#     mask = numpy.zeros((len(y), num_protos)).astype('float32')
#     for i, label in enumerate(y):
#         mask[i, label:label+1] = 1
#     return (mask,)

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

import theano
f = theano.function([x], out)

res = []
for x, m in train_stream.get_epoch_iterator():
    res.extend(f(x))
res = numpy.array(res)
C = numpy.cov(res, rowvar=0)
d, V = numpy.linalg.eig(C)
print d
print V.shape

W, = lvq.params
out2 = tensor.dot(out, V[:, :3])
prot = tensor.dot(W, V[:, :3])

D = ((prot**2).sum(axis=1, keepdims=True).T + (out2**2).sum(axis=1, keepdims=True) - 2*tensor.dot(out2, prot.T))
d_correct = (D + (1-mask)*numpy.float32(2e25)).min(axis=1)
d_incorrect = (D + mask*numpy.float32(2e25)).min(axis=1)

cost = ((d_correct - d_incorrect)/(d_correct+d_incorrect)).mean()
cost.name='cost'
misclass = (tensor.switch(d_correct - d_incorrect < 0, 0.0, 1.0).sum())/mask.shape[0]
misclass.name='misclass'

from blocks.monitoring.evaluators import DatasetEvaluator
ev = DatasetEvaluator([cost, misclass])
print ev.evaluate(test_stream)

