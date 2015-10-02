#######################
### Load model
#######################

from blocks.serialization import load
from blocks.filter import VariableFilter
from blocks.bricks import Softmax
from blocks.model import Model
from blocks.extensions.monitoring import DataStreamMonitoring

main_loop = load(open('./exp/lvq_2_bn_noise.pkl', 'rb'))

train_monitor = None
test_monitor = None
for ex in main_loop.extensions:
    if isinstance(ex, DataStreamMonitoring) and ex.prefix == 'train':
        train_monitor = ex
    if isinstance(ex, DataStreamMonitoring) and ex.prefix == 'test':
        test_monitor = ex

model = Model(test_monitor._evaluator.theano_variables[0])
y, x = model.inputs
D = VariableFilter(theano_name='test_D')(model.variables)[0]
out = VariableFilter(theano_name='mlp_inference_output')(model.variables)[0]
cost = VariableFilter(theano_name='test_cost')(model.variables)[0]
loss = model.outputs[0]
#loss, misclass = test_monitor._evaluator.theano_variables

#######################
### Noise
#######################

import theano
import theano.tensor as tensor
import numpy
from PIL import Image
from fuel.datasets import MNIST
from fuel.streams import DataStream
from fuel.schemes import ShuffledScheme

grad = tensor.grad(loss, x)

f_grad = theano.function([x, y], grad)
f_cost = theano.function([x, y], cost)
f_dist = theano.function([x], D)
f_emb = theano.function([x], out)

inp = numpy.random.uniform(low=0.0, high=1.0, size=(10000, 1, 28, 28)).astype('float32')
dist = f_dist(inp)
sorted_dist = numpy.sort(dist, axis=1)
conf = -1.0*(sorted_dist[:, 0] - sorted_dist[:, 1])/(sorted_dist[:, 0] + sorted_dist[:, 1])

test_stream = test_monitor.data_stream

print test_monitor._evaluator.evaluate(test_stream)

valid_losses = numpy.array([])
for x in test_stream.get_epoch_iterator():
    l = f_cost(x[0], x[1])
    valid_losses = numpy.append(valid_losses, -l)
    
print (valid_losses < -0.8).sum()

#valid_losses = valid_losses[valid_losses > -0.5]
from matplotlib import pyplot as plt
bins = numpy.linspace(-1.0, 1.0, 100)
plt.hist(conf, bins, alpha=0.5, label='noise')
#plt.hist(train_losses, bins, alpha=0.5, label='train')
plt.hist(valid_losses, bins, alpha=0.5, label='test')


##################
#### Adversarial
##################

epsilon = .25
num_adv = numpy.zeros((10,))
adv_confidences = numpy.array([])

for x in test_stream.get_epoch_iterator(as_dict=True):
    prediction = numpy.argmin(f_dist(x['features']), axis=1)
    
    g = f_grad(x['features'], x['targets'])
    g = numpy.sign(g)
    adv_x = x['features'] + epsilon*g
    
    adv_dist = f_dist(adv_x)
    adv_pred = numpy.argmin(adv_dist, axis=1)
    adv_sorted_dist = numpy.sort(adv_dist, axis=1)
    adv_conf = -1.0*(adv_sorted_dist[:, 0] - adv_sorted_dist[:, 1])/(adv_sorted_dist[:, 0] + adv_sorted_dist[:, 1])
    for p1, p2, conf in zip(prediction, adv_pred, adv_conf):
        if p1 != p2:
            adv_confidences = numpy.append(adv_confidences, -conf)
            num_adv[p1] += 1
        else:
            adv_confidences = numpy.append(adv_confidences, conf)
                
print num_adv
print numpy.sum(num_adv)
print numpy.mean(adv_confidences)

bins = numpy.linspace(-1.0, 1.0, 100)
plt.hist(adv_confidences, bins, alpha=0.5, label='adv')
plt.legend()
plt.savefig('figures/lvq_conf.pdf')
    
#############################
#### Embedding visualization
#############################

train_stream = train_monitor.data_stream

feature_embedding = []
labels = []
for x in train_stream.get_epoch_iterator(as_dict=True):
    feature_embedding.extend(f_emb(x['features']))
    labels.extend(x['targets'])

feature_embedding = numpy.array(feature_embedding)

C = numpy.cov(feature_embedding, rowvar=0)
d, V = numpy.linalg.eig(C)

fig = plt.figure()
plt.bar(range(100), d/d.sum())
plt.savefig('figures/lvq_embedding.pdf')


from matplotlib import pyplot as plt
fig = plt.figure()
from mpl_toolkits.mplot3d import axes3d, Axes3D
ax = Axes3D(fig)
features_pca = numpy.dot(feature_embedding, V[:, :3])
mc1 = [154/255., 106/255., 228/255.]
mc2 = [220/255., 170/255., 114/255.]
mc3 = [249/255., 85/255., 132/255.]

colors = ['b', 'c', 'm', 'k', 'r', 'g', 'y', mc1, mc2, mc3]
for i in range(10):
    data = numpy.asarray([x for x, y in zip(features_pca, labels) if y == i])
    ax.plot(data[:, 0], data[:, 1], data[:, 2], 'o', color=colors[i], label=str(i))
    
ax.legend()
plt.show()
