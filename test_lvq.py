import theano
import theano.tensor as tensor
import numpy

num_protos = 5
n_out= 5
n_examples = 5000
eta=0.2

h3 = tensor.fmatrix('x')
mask = tensor.fmatrix('mask')

W = theano.shared(numpy.random.normal(size=(num_protos, n_out)))
D = ((W**2).sum(axis=1, keepdims=True).T + (h3**2).sum(axis=1, keepdims=True) - 2*tensor.dot(h3, W.T))

d_correct = (D + (1-mask)*numpy.float32(1e30)).min(axis=1)
d_incorrect = (D + mask*numpy.float32(1e30)).min(axis=1)

mu = numpy.float32(0.2)
cost = (d_correct - d_incorrect) + 5.0*d_correct
misclass = (tensor.switch(d_correct - d_incorrect < 0, 0.0, 1.0).sum())/mask.shape[0]
loss = cost.mean()

params = [W]
grad = theano.grad(loss, params)
updates = [p-eta*g_p for p, g_p in zip(params, grad)]

f = theano.function([h3, mask], [loss, misclass], updates=zip(params, updates))
l = theano.function([h3, mask], [loss, misclass])

loc = -20
all_data = numpy.random.normal(loc=loc, size=(n_examples, n_out)).astype('float32')
all_mask = numpy.zeros((n_examples, num_protos), 'float32')
all_mask[:, 0] = 1

for c in range(1, 5):
    loc += 10
    all_data = numpy.vstack((all_data, numpy.random.normal(loc=loc, size=(n_examples, n_out)).astype('float32')))
    m = numpy.zeros((n_examples, num_protos), 'float32')
    m[:, c] = 1
    all_mask = numpy.vstack((all_mask, m))
    
ind = numpy.random.permutation(num_protos*n_examples)
data = all_data[ind, :]
mask = all_mask[ind, :]

batch_size= 200
n_batches = (num_protos*n_examples)/batch_size

for j in range(500):
    for i in range(n_batches):
        if i == 0:
            print f(data[i*batch_size:(i+1)*batch_size, :], mask[i*batch_size:(i+1)*batch_size, :])
        else:
            f(data[i*batch_size:(i+1)*batch_size, :], mask[i*batch_size:(i+1)*batch_size, :])
print W.get_value()








