#######################
### Load model
#######################

from blocks.serialization import load
from blocks.filter import VariableFilter
from blocks.bricks import Softmax

main_loop = load(open('./exp/lvq_deep.pkl', 'rb'))
model = main_loop.model

loss = VariableFilter(theano_name='lvq_apply_cost')(model.variables)[0]
x = VariableFilter(theano_name='features')(model.variables)[0]
mask = VariableFilter(theano_name='mask')(model.variables)[0]
pred = VariableFilter(theano_name='D')(model.variables)[0]

#######################
### Adversarial 
#######################

import theano
import theano.tensor as tensor
import numpy
from PIL import Image
from mnist import MNIST
from fuel.streams import DataStream
from fuel.schemes import ShuffledScheme

grad = tensor.grad(loss, x)

f_grad = theano.function([x, mask], grad)
f_pred = theano.function([x, mask], [loss, pred])
f_pred2 = theano.function([x], pred)

inp = numpy.random.uniform(low=0.0, high=1.0, size=(1000, 784)).astype('float32')
dist = f_pred2(inp)
sorted_dist = numpy.sort(dist, axis=1)
print sorted_dist[0, :]
c = (sorted_dist[:, 0] - sorted_dist[:, 1])/(sorted_dist[:, 0] + sorted_dist[:, 1])
print (c < -0.8).sum()

batch_size = 1
mnist_test = MNIST('test', n_protos=1, drop_input=False, sources=('features', 'mask'))
test_stream = DataStream(mnist_test, iteration_scheme=ShuffledScheme(mnist_test.num_examples, batch_size))

from blocks.monitoring.evaluators import DatasetEvaluator
ev = DatasetEvaluator([loss])
print ev.evaluate(test_stream)

epsilon = 1
num_adv = 0
c = 0.0
c_new = 0.0
j=0


for x in test_stream.get_epoch_iterator():
    if j==5:
        cost, pred = f_pred(x[0], x[1])
        print pred
        pred = pred.argmin(axis=1)
        print pred[0]
        print cost
        I = numpy.reshape(x[0][0, :], (28, 28))
        I8 = (((I - I.min()) / (I.max() - I.min())) * 255.9).astype(numpy.uint8)
        im = Image.fromarray(I8)
        im.save('im0.png')
        new_x = x[0] #numpy.random.uniform(size=(1, 784), low=0, high=1.0).astype('float32')
        for i in range(1, 30):
            g = f_grad(new_x, x[1])
            g = g/numpy.linalg.norm(g)
            new_x = new_x + epsilon*g
	    new_x = numpy.minimum(numpy.maximum(new_x, 0.0), 1.0)
	    print new_x.max()
            cost, pred = f_pred(new_x, x[1])
            print pred
            print cost
            I = numpy.reshape(new_x[0], (28, 28))
            I8 = (((I - I.min()) / (I.max() - I.min())) * 255.9).astype(numpy.uint8)
            im = Image.fromarray(I8)
            im.save('im'+str(i)+'.png')
        
        break
    j+=1
    
    
    
    




