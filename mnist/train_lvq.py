def main(n_hiddens=1500, n_out=200, noise_std=0.5,
         learning_rate=2e-1, momentum=0.9, gamma=2.0):
             
    ######################
    # Model 
    ######################
    import theano
    from theano.sandbox.rng_mrg import MRG_RandomStreams
    import numpy
    import theano.tensor as tensor
    from lvq.lvq import AaronLVQ, SupervisedNG, initialize_prototypes

    from blocks.bricks import Linear, LinearMaxout
    from blocks.bricks import Rectifier, Softmax, Logistic
    from blocks.bricks.cost import MisclassificationRate
    from blocks.config import config
    from blocks.initialization import Uniform, Constant
    from blocks.model import Model

    from lvq.batch_norm import MLP
    #from blocks.bricks import MLP

    seed = config.default_seed
    rng = MRG_RandomStreams(seed)

    x = tensor.tensor4('features')
    flat_x = tensor.flatten(x, outdim=2)
    flat_x_noise = flat_x + rng.normal(size=flat_x.shape, std=noise_std)
    y = tensor.imatrix('targets')
    flat_y = tensor.flatten(y, outdim=1)

    act = Rectifier() 
    mlp = MLP(dims=[784, n_hiddens, n_hiddens, n_out], activations=[act, act, None])
    mlp.weights_init = Uniform(0.0, 0.001)
    mlp.biases_init = Constant(0.0)
    mlp.initialize()

    train_out = mlp.apply(flat_x_noise)
    test_out = mlp.inference(flat_x)

    lamb = theano.shared(numpy.float32(10.0))
    lvq = SupervisedNG(10, n_out, nonlin=True, gamma=gamma, lamb=lamb, name='lvq')

    test_loss, test_misclass = lvq.apply(test_out, flat_y, 'test')
    loss, misclass = lvq.apply(train_out, flat_y, 'train')

    model = Model(loss)

    ######################
    # Data
    ######################

    #from mnist import MNIST
    from fuel.datasets import MNIST
    from fuel.streams import DataStream
    from fuel.schemes import ShuffledScheme, SequentialScheme
    from fuel.transformers import ScaleAndShift, ForceFloatX

    mnist_train = MNIST(['train']) #MNIST('train', drop_input=False, sources=('features', 'targets'))
    mnist_test = MNIST(['test']) # MNIST('test', drop_input=False, sources=('features', 'targets'))

    batch_size = 100 #Batch size for training
    batch_size_mon = 2000 # Batch size for monitoring and batch normalization
    n_batches = int(numpy.ceil(float(mnist_train.num_examples)/batch_size_mon))
    num_protos = 10
    
    ind = range(mnist_train.num_examples)
    train_ind = ind[:50000]
    val_ind = ind[50000:]

    def preprocessing(data_stream):
        return ForceFloatX(ScaleAndShift(data_stream, 1/255.0, 0.0, which_sources=('features',)), which_sources=('features',))

    train_stream_mon = preprocessing(DataStream(mnist_train, iteration_scheme=ShuffledScheme(train_ind, batch_size_mon)))
    train_stream_bn = preprocessing(DataStream(mnist_train, iteration_scheme=ShuffledScheme(train_ind, batch_size_mon)))
    train_stream = preprocessing(DataStream(mnist_train, iteration_scheme=ShuffledScheme(train_ind, batch_size)))
    valid_stream =  preprocessing(DataStream(mnist_train, iteration_scheme=ShuffledScheme(val_ind, batch_size)))
    test_stream = preprocessing(DataStream(mnist_test, iteration_scheme=ShuffledScheme(mnist_test.num_examples, batch_size_mon)))

    initialize_prototypes(lvq, x, train_out, train_stream_mon)

    ######################
    # Training
    ######################

    from blocks.main_loop import MainLoop
    from blocks.extensions import FinishAfter,Timing, Printing
    from blocks.extensions.monitoring import DataStreamMonitoring, TrainingDataMonitoring
    from blocks.algorithms import GradientDescent, Momentum, RMSProp, Adam
    from blocks.extensions.saveload import Checkpoint
    from lvq.extensions import EarlyStopping, LRDecay, MomentumSwitchOff, NCScheduler

    from lvq.batch_norm import BatchNormExtension
    lr = theano.shared(numpy.float32(1e-3))
    step_rule = Momentum(learning_rate, 0.9) #Adam(learning_rate=lr) #RMSProp(learning_rate=1e-5, max_scaling=1e4)
    num_epochs = 100

    earlystop = EarlyStopping('valid_lvq_apply_misclass', 100, './exp/alvq.pkl')
    
    main_loop = MainLoop(
        model=model,
        data_stream=train_stream,
        algorithm=GradientDescent(
            cost=model.outputs[0], parameters=model.parameters, step_rule=step_rule),
        extensions=[FinishAfter(after_n_epochs=num_epochs),
                    BatchNormExtension(model, train_stream_bn, n_batches),
                    LRDecay(step_rule.learning_rate, [20, 40, 60, 80]), 
                    MomentumSwitchOff(step_rule.momentum, num_epochs-20),
                    NCScheduler(lamb, 30., 0, num_epochs),
                    DataStreamMonitoring(
                        variables=[test_loss, test_misclass],
                        data_stream=train_stream_mon,
                        prefix='train'),
                    DataStreamMonitoring(
                        variables=[test_loss, test_misclass],
                        data_stream=valid_stream,
                        prefix='valid'),
                    DataStreamMonitoring(
                        variables=[test_loss, test_misclass],
                        data_stream=test_stream,
                        prefix='test'),
                    Timing(), 
                    earlystop,
                    Printing(after_epoch=True)])
    main_loop.run()
    
    return main_loop.status.get('best_test_lvq_apply_misclass', None)

if __name__ == '__main__':
    print main()