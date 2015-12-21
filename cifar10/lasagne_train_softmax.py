import theano
import numpy
import theano.tensor as tensor
import lasagne
import time
from batch_norm import batch_norm

def load_data():
    import cPickle
    d = cPickle.load(open("/Tmp/devries/cifar10_raw.pkl", 'rb')) 
    train_x = d['train_x'][:45000]
    train_y = d['train_y'][:45000]
    valid_x = d['train_x'][45000:]
    valid_y = d['train_y'][45000:]
    return train_x, train_y, valid_x, valid_y, d['test_x'], d['test_y']

#def build_model(input_var):
    #network = lasagne.layers.InputLayer(shape=(None, 3, 32, 32),
                                        #input_var=input_var)
    #network = batch_norm(lasagne.layers.Conv2DLayer(
            #network, num_filters=64, filter_size=(3, 3),
            #nonlinearity=lasagne.nonlinearities.rectify,
            #W=lasagne.init.HeNormal(), pad='same'))
    #network = lasagne.layers.dropout(network, p=0.3)
    #network = batch_norm(lasagne.layers.Conv2DLayer(
            #network, num_filters=64, filter_size=(3, 3),
            #nonlinearity=lasagne.nonlinearities.rectify,
            #W=lasagne.init.HeNormal(), pad='same'))
    #network = lasagne.layers.MaxPool2DLayer(network, pool_size=(2, 2))

    #network = batch_norm(lasagne.layers.Conv2DLayer(
            #network, num_filters=128, filter_size=(3, 3),
            #nonlinearity=lasagne.nonlinearities.rectify,
            #W=lasagne.init.HeNormal(), pad='same'))
    #network = lasagne.layers.dropout(network, p=0.4)
    #network = batch_norm(lasagne.layers.Conv2DLayer(
            #network, num_filters=128, filter_size=(3, 3),
            #nonlinearity=lasagne.nonlinearities.rectify,
            #W=lasagne.init.HeNormal(), pad='same'))        
    #network = lasagne.layers.MaxPool2DLayer(network, pool_size=(2, 2))
            
    #network = batch_norm(lasagne.layers.Conv2DLayer(
            #network, num_filters=256, filter_size=(3, 3),
            #nonlinearity=lasagne.nonlinearities.rectify,
            #W=lasagne.init.HeNormal(), pad='same'))
    #network = lasagne.layers.dropout(network, p=0.4)
    #network = batch_norm(lasagne.layers.Conv2DLayer(
            #network, num_filters=256, filter_size=(3, 3),
            #nonlinearity=lasagne.nonlinearities.rectify,
            #W=lasagne.init.HeNormal(), pad='same'))
    #network = lasagne.layers.dropout(network, p=0.4)
    #network = batch_norm(lasagne.layers.Conv2DLayer(
            #network, num_filters=256, filter_size=(3, 3),
            #nonlinearity=lasagne.nonlinearities.rectify,
            #W=lasagne.init.HeNormal(), pad='same'))
    #network = lasagne.layers.MaxPool2DLayer(network, pool_size=(2, 2))
    
    #network = batch_norm(lasagne.layers.Conv2DLayer(
            #network, num_filters=512, filter_size=(3, 3),
            #nonlinearity=lasagne.nonlinearities.rectify,
            #W=lasagne.init.HeNormal(), pad='same'))
    #network = lasagne.layers.dropout(network, p=0.4)
    #network = batch_norm(lasagne.layers.Conv2DLayer(
            #network, num_filters=512, filter_size=(3, 3),
            #nonlinearity=lasagne.nonlinearities.rectify,
            #W=lasagne.init.HeNormal(), pad='same'))
    #network = lasagne.layers.dropout(network, p=0.4)
    #network = batch_norm(lasagne.layers.Conv2DLayer(
            #network, num_filters=512, filter_size=(3, 3),
            #nonlinearity=lasagne.nonlinearities.rectify,
            #W=lasagne.init.HeNormal(), pad='same'))
    #network = lasagne.layers.MaxPool2DLayer(network, pool_size=(2, 2))
    
    #network = batch_norm(lasagne.layers.Conv2DLayer(
            #network, num_filters=512, filter_size=(3, 3),
            #nonlinearity=lasagne.nonlinearities.rectify,
            #W=lasagne.init.HeNormal(), pad='same'))
    #network = lasagne.layers.dropout(network, p=0.4)
    #network = batch_norm(lasagne.layers.Conv2DLayer(
            #network, num_filters=512, filter_size=(3, 3),
            #nonlinearity=lasagne.nonlinearities.rectify,
            #W=lasagne.init.HeNormal(), pad='same'))
    #network = lasagne.layers.dropout(network, p=0.4)
    #network = batch_norm(lasagne.layers.Conv2DLayer(
            #network, num_filters=512, filter_size=(3, 3),
            #nonlinearity=lasagne.nonlinearities.rectify,
            #W=lasagne.init.HeNormal(), pad='same'))
    #network = lasagne.layers.MaxPool2DLayer(network, pool_size=(2, 2))
    #network = lasagne.layers.dropout(network, p=0.5)
    #network = batch_norm(lasagne.layers.DenseLayer(
            #network, num_units=512,
            #nonlinearity=lasagne.nonlinearities.rectify, 
            #W=lasagne.init.HeNormal()))
       
    #network = lasagne.layers.dropout(network, p=0.5)
    #network = lasagne.layers.DenseLayer(network,
            #num_units=10, W=lasagne.init.HeNormal(), 
            #nonlinearity=None)
    
    #return network

def build_model(input_var):
    network = lasagne.layers.InputLayer(shape=(None, 3, 32, 32),
                                        input_var=input_var)
    network = batch_norm(lasagne.layers.Conv2DLayer(
            network, num_filters=96, filter_size=(3, 3),
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.HeNormal(), pad='same'))
    #network = lasagne.layers.dropout(network, p=0.4)
    network = batch_norm(lasagne.layers.Conv2DLayer(
            network, num_filters=96, filter_size=(3, 3),
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.HeNormal(), pad='same'))
    network = lasagne.layers.MaxPool2DLayer(network, pool_size=(2, 2))

    network = batch_norm(lasagne.layers.Conv2DLayer(
            network, num_filters=192, filter_size=(3, 3),
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.HeNormal(), pad='same'))
    #network = lasagne.layers.dropout(network, p=0.4)
    network = batch_norm(lasagne.layers.Conv2DLayer(
            network, num_filters=192, filter_size=(3, 3),
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.HeNormal(), pad='same'))        
    network = lasagne.layers.MaxPool2DLayer(network, pool_size=(2, 2))
            
    network = batch_norm(lasagne.layers.Conv2DLayer(
            network, num_filters=256, filter_size=(3, 3),
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.HeNormal(), pad='same'))
    #network = lasagne.layers.dropout(network, p=0.4)
    network = batch_norm(lasagne.layers.Conv2DLayer(
            network, num_filters=256, filter_size=(3, 3),
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.HeNormal(), pad='same'))
    #network = lasagne.layers.dropout(network, p=0.4)
    #network = batch_norm(lasagne.layers.Conv2DLayer(
            #network, num_filters=256, filter_size=(3, 3),
            #nonlinearity=lasagne.nonlinearities.rectify,
            #W=lasagne.init.HeNormal(), pad='same'))
    network = lasagne.layers.MaxPool2DLayer(network, pool_size=(2, 2))
    
    network = batch_norm(lasagne.layers.Conv2DLayer(
            network, num_filters=512, filter_size=(3, 3),
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.HeNormal(), pad='same'))
    #network = lasagne.layers.dropout(network, p=0.4)
    network = batch_norm(lasagne.layers.Conv2DLayer(
            network, num_filters=512, filter_size=(3, 3),
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.HeNormal(), pad='same'))
    #network = lasagne.layers.dropout(network, p=0.4)
    #network = batch_norm(lasagne.layers.Conv2DLayer(
            #network, num_filters=512, filter_size=(3, 3),
            #nonlinearity=lasagne.nonlinearities.rectify,
            #W=lasagne.init.HeNormal(), pad='same'))
    network = lasagne.layers.MaxPool2DLayer(network, pool_size=(2, 2))
    
    #network = batch_norm(lasagne.layers.Conv2DLayer(
            #network, num_filters=512, filter_size=(3, 3),
            #nonlinearity=lasagne.nonlinearities.rectify,
            #W=lasagne.init.HeNormal(), pad='same'))
    ##network = lasagne.layers.dropout(network, p=0.4)
    #network = batch_norm(lasagne.layers.Conv2DLayer(
            #network, num_filters=512, filter_size=(3, 3),
            #nonlinearity=lasagne.nonlinearities.rectify,
            #W=lasagne.init.HeNormal(), pad='same'))
    ##network = lasagne.layers.dropout(network, p=0.4)
    #network = batch_norm(lasagne.layers.Conv2DLayer(
            #network, num_filters=512, filter_size=(3, 3),
            #nonlinearity=lasagne.nonlinearities.rectify,
            #W=lasagne.init.HeNormal(), pad='same'))
    #network = lasagne.layers.MaxPool2DLayer(network, pool_size=(2, 2))
    ##network = lasagne.layers.dropout(network, p=0.5)
    #network = batch_norm(lasagne.layers.DenseLayer(
            #network, num_units=512,
            #nonlinearity=lasagne.nonlinearities.rectify, 
            #W=lasagne.init.HeNormal()))
       
    #network = lasagne.layers.dropout(network, p=0.5)
    network = lasagne.layers.DenseLayer(network,
            num_units=200, W=lasagne.init.HeNormal(), 
            nonlinearity=lasagne.nonlinearities.rectify)
    
    network = lasagne.layers.DenseLayer(network, num_units=10, 
                                        W=lasagne.init.HeNormal(), 
                                        nonlinearity=None)
    
    return network

def iterate_minibatches(inputs, targets, batchsize, shuffle=False):
    assert len(inputs) == len(targets)
    if shuffle:
        indices = numpy.arange(len(inputs))
        numpy.random.shuffle(indices)
    for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
        yield inputs[excerpt], targets[excerpt]

def horizontal_flip(input):
    new_inputs = []
    for x in input:
        if numpy.random.randint(2) == 1:
            new_inputs.append(x[:, :, ::-1])
        else:
            new_inputs.append(x)
    return numpy.array(new_inputs)

def main(num_epochs=300, lr_start=0.1, lr_end=1e-6, momentum=0.9, optimizer='momentum'):
    X_train, y_train, X_val, y_val, X_test, y_test = load_data() 

    x = tensor.tensor4('x')
    y = tensor.imatrix('y')
    flat_y = tensor.flatten(y, outdim=1)

    network = build_model(x)

    train_out = lasagne.layers.get_output(network)
    train_prob = lasagne.nonlinearities.softmax(train_out)
    train_nll = lasagne.objectives.categorical_crossentropy(train_prob, flat_y).mean()
    train_misclass = 1.0 - lasagne.objectives.categorical_accuracy(train_prob, flat_y).mean()

    test_out = lasagne.layers.get_output(network, deterministic=True)
    test_prob = lasagne.nonlinearities.softmax(test_out)
    test_nll = lasagne.objectives.categorical_crossentropy(test_prob, flat_y).mean()
    test_misclass = 1.0 - lasagne.objectives.categorical_accuracy(test_prob, flat_y).mean()

    reg = lasagne.regularization.regularize_network_params(network, lasagne.regularization.l2)*0.0005
    layers = lasagne.layers.get_all_layers(network)
    for l in layers:
        print l.output_shape

    params = lasagne.layers.get_all_params(network, trainable=True)
    lr_decay = (lr_end/lr_start)**(1./num_epochs)
    lr = theano.shared(numpy.float32(lr_start))
    if optimizer == 'adam':
        updates = lasagne.updates.adam(train_nll+reg, params, learning_rate=lr)
    else:
        updates = lasagne.updates.momentum(train_nll+reg, params, lr, momentum)
        
    train_fn = theano.function([x, y], [train_nll, train_misclass], updates=updates)
    val_fn = theano.function([x, y], [test_nll, test_misclass])
    
    best_val_err = None
    print("Starting training...")
    
    # We iterate over epochs:
    for epoch in range(num_epochs):
        # In each epoch, we do a full pass over the training data:
        lr.set_value(numpy.float32(lr.get_value()*lr_decay))
            
        train_nll = 0.0    
        train_err = 0.0
        train_batches = 0
        start_time = time.time()
        for batch in iterate_minibatches(X_train, y_train, 100, shuffle=True):
            inputs, targets = batch
            nll, err = train_fn(horizontal_flip(inputs), targets)
            train_nll += nll
            train_err += err
            train_batches += 1

        # And a full pass over the validation data:
        val_nll = 0.0
        val_err = 0.0
        val_batches = 0
        for batch in iterate_minibatches(X_val, y_val, 500, shuffle=False):
            inputs, targets = batch
            nll, err = val_fn(inputs, targets)
            val_nll += nll
            val_err += err 
            val_batches += 1

        # Then we print the results for this epoch:
        print("Epoch {} of {} took {:.3f}s".format(
            epoch + 1, num_epochs, time.time() - start_time))
        print("  training loss:\t\t{:.6f}".format(train_nll / train_batches))
        print("  training error:\t\t{:.6f}".format(train_err*100 / train_batches))
        print("  validation loss:\t\t{:.6f}".format(val_nll / val_batches))
        print("  validation error:\t\t{:.6f} %".format(val_err*100 / val_batches))
        
        if (best_val_err is None or val_err <= best_val_err):
            best_val_err = val_err
            numpy.savez('/Tmp/devries/softmax.npz', *lasagne.layers.get_all_param_values(network))
        
if __name__ == '__main__':
	main()
