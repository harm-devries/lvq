#######################
### Load model
#######################
import theano.tensor as tensor
from blocks.serialization import load
from blocks.model import Model
from blocks.filter import VariableFilter
from blocks.bricks import Softmax
from blocks.bricks.cost import MisclassificationRate
from blocks.extensions.monitoring import DataStreamMonitoring

main_loop = load(open('./exp/softmax_2_bn_noise.pkl', 'rb'))

train_monitor = None
test_monitor = None
for ex in main_loop.extensions:
    if isinstance(ex, DataStreamMonitoring) and ex.prefix == 'train':
        train_monitor = ex
    if isinstance(ex, DataStreamMonitoring) and ex.prefix == 'test':
        test_monitor = ex
        
model = Model(test_monitor._evaluator.theano_variables[0])

loss = model.outputs[0]
y, x = model.inputs
out = VariableFilter(theano_name='mlp_inference_output')(model.variables)[0]
out2 = VariableFilter(theano_name='linear_apply_output')(model.variables)[0]
pred = Softmax().apply(out2)
misclass = MisclassificationRate().apply(tensor.flatten(y, outdim=1), out)

##############
### Noise
##############

import theano
import theano.tensor as tensor
import numpy
from PIL import Image
#from mnist import MNIST
from fuel.datasets import MNIST
from fuel.streams import DataStream
from fuel.schemes import ShuffledScheme

grad = tensor.grad(loss, x)

f_grad = theano.function([x, y], grad)
f_pred = theano.function([x], pred)
f_emb = theano.function([x], out)

inp = numpy.random.uniform(low=0.0, high=1.0, size=(10000, 1, 28, 28)).astype('float32')
p = f_pred(inp).max(axis=1)

test_stream = test_monitor.data_stream
print test_monitor._evaluator.evaluate(test_stream)

test_losses = numpy.array([])
for x in test_stream.get_epoch_iterator():
    l = f_pred(x[0]).max(axis=1)
    test_losses = numpy.append(test_losses, l)

#valid_losses = valid_losses[valid_losses > -0.5]
from matplotlib import pyplot as plt
bins = numpy.linspace(-1.0, 1.0, 100)
plt.hist(p, bins, alpha=0.5, label='noise')
#plt.hist(train_losses, bins, alpha=0.5, label='train')
plt.hist(test_losses, bins, alpha=0.5, label='test')


########################
#### Adversarial 
########################

epsilon = 0.25
num_adv = numpy.zeros((10,))
adv_confidences = numpy.array([])

for x in test_stream.get_epoch_iterator(as_dict=True):
    sm = f_pred(x['features'])
    prediction = numpy.argmax(sm, axis=1)
    g = f_grad(x['features'], x['targets'])
    g = numpy.sign(g)
    #print x['features'].shape
    #print g.shape
    new_x = x['features'] + epsilon*g
    
    #I = x['features'][0, 0, :, :]
    #im = Image.fromarray(I)
    #im.save('im.png')
    #I = new_x[0, 0, :, :]
    #im = Image.fromarray(I)
    #im.save('new_im.png')
    adv_sm = f_pred(new_x)
    adv_prediction = numpy.argmax(adv_sm, axis=1)
    adv_conf = numpy.max(adv_sm, axis=1)
    for p1, p2, conf in zip(prediction, adv_prediction, adv_conf):
        if p1 != p2:
            num_adv[p1] += 1
            adv_confidences = numpy.append(adv_confidences, -conf)
        else:
            adv_confidences = numpy.append(adv_confidences, conf)
    
print num_adv
print numpy.sum(num_adv)
plt.hist(adv_confidences, bins, alpha=0.5, label='adv')
print numpy.mean(adv_confidences)
plt.legend()
plt.savefig('figures/softmax_conf.pdf')    
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
plt.savefig('figures/softmax_embedding.pdf')

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
    




