import numpy
import theano
import theano.tensor as tensor

######################
# Model 
######################

def main(n_inputs=784,
         n_hiddens0=1000,
         n_hiddens1=1000,
         n_out=10,
         n_proto_class=1,
         batch_size=1,
         eta=numpy.float32(0.001)):
             
    def rectifier(x):
        return tensor.switch(x > 0, x, 0)
        
    def init_param(n_x, n_y):
        return theano.shared(numpy.random.normal(size=(n_x, n_y)).astype('float32'))
        
    W0 = init_param(n_inputs, n_hiddens0)
    W1 = init_param(n_hiddens0, n_hiddens1)
    W2 = init_param(n_hiddens1, n_out)
    b0 = theano.shared(numpy.zeros(n_hiddens0, 'float32'))
    b1 = theano.shared(numpy.zeros(n_hiddens1,'float32'))
    b2 = theano.shared(numpy.zeros(n_out, 'float32'))

    x = tensor.fmatrix('x')
    mask = tensor.fmatrix('mask')
    
    num_protos = 10*n_proto_class
    W = init_param(num_protos, n_out)
    
    h1 = rectifier(tensor.dot(x, W0) + b0)
    h2 = rectifier(tensor.dot(h1, W1) + b1)
    h3 = rectifier(tensor.dot(h2, W2) + b2)
    
    W = theano.shared(numpy.random.normal(size=(num_protos, n_out)))

    D = ((W**2).sum(axis=1, keepdims=True).T + (h3**2).sum(axis=1, keepdims=True) - 2*tensor.dot(h3, W.T))
    d_correct = (D + (1-mask)*numpy.float32(2e25)).min(axis=1)
    d_incorrect = (D + mask*numpy.float32(2e25)).min(axis=1)
    
    mu = numpy.float32(1.0)
    cost = ((d_correct - d_incorrect)/(d_correct+d_incorrect))
    misclass = tensor.switch(cost < 0, 0.0, 1.0).sum()/mask.shape[0]
    loss = cost.mean()
    
    params = [W, W0, W1, W2, b0, b1, b2]
    grad = theano.grad(loss, params)
    
    updates = [p-eta*g_p for p, g_p in zip(params, grad)]
    f = theano.function([x, mask], loss, updates=zip(params, updates))
    l = theano.function([x, mask], [loss, misclass])
    
    from fuel.datasets.mnist import MNIST
    from fuel.streams import DataStream
    from fuel.schemes import ShuffledScheme

    mnist_train = MNIST(['train'])
    mnist_valid = MNIST(['test'])

    train_stream = DataStream(mnist_train, iteration_scheme=ShuffledScheme(mnist_train.num_examples, batch_size))
    valid_stream = DataStream(mnist_valid, iteration_scheme=ShuffledScheme(mnist_valid.num_examples, batch_size))
    
    def get_error(stream):
        loss = 0.0
        misclass = 0.0
        i = 0.0
        for x, y in stream.get_epoch_iterator():
            mask = numpy.zeros((len(y), num_protos)).astype('float32')
            for i, label in enumerate(y):
                mask[i, label:label+n_proto_class] = 1
            x = x.reshape((x.shape[0], n_inputs)).astype('float32')
            c_l, c_m = l(x, mask)
            loss += len(y)*c_l
            misclass += len(y)*c_m
            i+=len(y)
        print i
        return loss/i, misclass/i
    
    for epoch in range(50):
        for x, y in train_stream.get_epoch_iterator():
            mask = numpy.zeros((len(y), num_protos)).astype('float32')
            for i, label in enumerate(y):
                mask[i, label:label+n_proto_class] = 1
            x = x.reshape((x.shape[0], n_inputs)).astype('float32')
            f(x, mask)
            
        print get_error(train_stream), get_error(valid_stream)
        
if __name__ == '__main__':
    main()
        
        
    
    


