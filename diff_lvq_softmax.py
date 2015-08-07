import theano
import theano.tensor as tensor
import numpy

num_protos = 3
n_out= 2
n_examples = 500
eta=0.2

h3 = tensor.fmatrix('x')
mask = tensor.fmatrix('mask')

W = theano.shared(numpy.random.normal(size=(num_protos, n_out)))
D = ((W**2).sum(axis=1, keepdims=True).T + (h3**2).sum(axis=1, keepdims=True) - 2*tensor.dot(h3, W.T))

d_correct = (D + (1-mask)*numpy.float32(1e30)).min(axis=1)
d_incorrect = (D + mask*numpy.float32(1e30)).min(axis=1)

mu = numpy.float32(0.2)
cost = (d_correct - d_incorrect)/(d_correct + d_incorrect)
misclass = (tensor.switch(d_correct - d_incorrect < 0, 0.0, 1.0).sum())/mask.shape[0]
loss = cost.mean()

params = [W]
grad = theano.grad(loss, params)
updates = [p-eta*g_p for p, g_p in zip(params, grad)]

f = theano.function([h3, mask], [loss, misclass], updates=zip(params, updates))
l = theano.function([h3, mask], [loss, misclass])
f_D = theano.function([h3], D)

d1 = numpy.random.multivariate_normal(mean=(0, -6), cov=numpy.eye(2), size=n_examples).astype('float32')
m1 = numpy.zeros((n_examples, num_protos), 'float32')
m1[:, 0] = 1

d2 = numpy.random.multivariate_normal(mean=(-3, 0), cov=numpy.eye(2), size=n_examples).astype('float32')
m2 = numpy.zeros((n_examples, num_protos), 'float32')
m2[:, 1] = 1

d3 = numpy.random.multivariate_normal(mean=(3, 0), cov=numpy.eye(2), size=n_examples).astype('float32')
m3 = numpy.zeros((n_examples, num_protos), 'float32')
m3[:, 2] = 1

all_labels = numpy.concatenate((numpy.zeros((n_examples,), 'int32'), numpy.ones((n_examples,), 'int32'), numpy.ones((n_examples,), 'int32')*2))
all_data = numpy.vstack((d1, d2, d3))    
all_mask = numpy.vstack((m1, m2, m3))
    
ind = numpy.random.permutation(num_protos*n_examples)
data = all_data[ind, :]
mask = all_mask[ind, :]
labels = all_labels[ind]

batch_size= 200
n_batches = (num_protos*n_examples)/batch_size
for j in range(100):
    for i in range(n_batches):
        if i == 0:
            print f(data[i*batch_size:(i+1)*batch_size, :], mask[i*batch_size:(i+1)*batch_size, :])
        else:
            f(data[i*batch_size:(i+1)*batch_size, :], mask[i*batch_size:(i+1)*batch_size, :])


import matplotlib.pyplot as plt
from matplotlib.axes import Axes
import matplotlib.cm as cm
from scipy.spatial import voronoi_plot_2d, Voronoi

weights = W.get_value()

vor = Voronoi(weights)
voronoi_plot_2d(vor)

plt.plot(d1[:, 0], d1[:, 1], 'ro', markersize=3)
plt.plot(d2[:, 0], d2[:, 1], 'bo', markersize=3)
plt.plot(d3[:, 0], d3[:, 1], 'go', markersize=3)

plt.plot(weights[0, 0], weights[0, 1], 'r*', markersize=20)
plt.plot(weights[1, 0], weights[1, 1], 'b*', markersize=20)
plt.plot(weights[2, 0], weights[2, 1], 'g*', markersize=20)

X = numpy.arange(-10, 10, 0.1)
Y = numpy.arange(-10, 10, 0.1)
Z = numpy.zeros((len(X),len(Y)))

for i in range(len(X)):
    for j in range(len(Y)):
        inp = numpy.array([[X[i], Y[j]]]).astype('float32')
        d = numpy.sort(f_D(inp))
        Z[j, i] = (d[0][0]- d[0][1])/(d[0][0] + d[0][1])

im = plt.imshow(Z[:, :], interpolation='bilinear', origin='lower',
                cmap=cm.gray, extent=(-10, 10, -10, 10))
plt.colorbar(im, orientation='horizontal', shrink=0.8)

plt.xlim((-10, 10))
plt.ylim((-10, 10))
plt.savefig('lvq.pdf')


###################
#### Softmax
###################

from blocks.bricks import Softmax
from blocks.bricks.cost import MisclassificationRate

W2 = theano.shared(numpy.random.normal(size=(n_out, num_protos)).astype('float32'))
b = theano.shared(numpy.zeros((num_protos,)).astype('float32'))
y = tensor.ivector('y')

h = tensor.dot(h3, W2) + b
sm = Softmax()
pred = sm.apply(h)
misclass = MisclassificationRate().apply(y, pred)
c = sm.categorical_cross_entropy(y, h)

s_params = [W2, b]
s_grad = theano.grad(c, s_params)
s_updates = [p - numpy.float32(0.05)*g for p, g in zip(s_params, s_grad)]
s_f = theano.function([h3, y], [c, misclass], updates=zip(s_params, s_updates))
s_pred = theano.function([h3], pred)

for j in range(200):
    for i in range(n_batches):
	if i == 0:
            print s_f(data[i*batch_size:(i+1)*batch_size, :], labels[i*batch_size:(i+1)*batch_size])
	else:
            s_f(data[i*batch_size:(i+1)*batch_size, :], labels[i*batch_size:(i+1)*batch_size])

fig1 = plt.figure()

plt.plot(d1[:, 0], d1[:, 1], 'ro', markersize=3)
plt.plot(d2[:, 0], d2[:, 1], 'bo', markersize=3)
plt.plot(d3[:, 0], d3[:, 1], 'go', markersize=3)

for i in range(len(X)):
    for j in range(len(Y)):
        inp = numpy.array([[X[i], Y[j]]]).astype('float32')
        Z[j, i] = s_pred(inp).max()

im = plt.imshow(Z[:, :], interpolation='bilinear', origin='lower',
                cmap=cm.gray, extent=(-10, 10, -10, 10))
plt.colorbar(im, orientation='horizontal', shrink=0.8)

plt.xlim((-10, 10))
plt.ylim((-10, 10))
plt.savefig('softmax.pdf')

