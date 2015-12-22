import numpy
import lasagne
from lasagne_train_softmax import build_model, load_data, iterate_minibatches
import theano
import theano.tensor as tensor

x = tensor.tensor4('x')
y = tensor.imatrix('y')
flat_y = tensor.flatten(y, outdim=1)

network = build_model(x)

test_out = lasagne.layers.get_output(network, deterministic=True)
test_prob = lasagne.nonlinearities.softmax(test_out)
test_nll = lasagne.objectives.categorical_crossentropy(test_prob, flat_y).mean()
test_misclass = 1.0 - lasagne.objectives.categorical_accuracy(test_prob, flat_y).mean()

with numpy.load('/Tmp/devries/softmax.npz') as f:
    param_values = [f['arr_%d' % i] for i in range(len(f.files))]
    lasagne.layers.set_all_param_values(network, param_values)

X_train, y_train, X_val, y_val, X_test, y_test = load_data()
    
val_fn = theano.function([x, y], [test_nll, test_misclass])

val_nll = 0.0
val_err = 0.0
val_batches = 0
for batch in iterate_minibatches(X_train, y_train, 500, shuffle=False):
    inputs, targets = batch
    nll, err = val_fn(inputs, targets)
    val_nll += nll
    val_err += err 
    val_batches += 1

print("  validation loss:\t\t{:.6f}".format(val_nll / val_batches))
print("  validation error:\t\t{:.6f} %".format(val_err*100 / val_batches))