import theano
from blocks.bricks import Initializable, Feedforward
from blocks.bricks.conv import Convolutional, MaxPooling
from batch_norm import BatchNorm
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

from blocks.bricks.base import application, lazy

class ConvolutionalSequence(Initializable, Feedforward):
    """A sequence of convolutional operations.
    Parameters
    ----------
    layers : list
        List of convolutional bricks (i.e. :class:`ConvolutionalActivation`
        or :class:`ConvolutionalLayer`)
    num_channels : int
        Number of input channels in the image. For the first layer this is
        normally 1 for grayscale images and 3 for color (RGB) images. For
        subsequent layers this is equal to the number of filters output by
        the previous convolutional layer.
    batch_size : int, optional
        Number of images in batch. If given, will be passed to
        theano's convolution operator resulting in possibly faster
        execution.
    image_size : tuple, optional
        Width and height of the input (image/featuremap). If given,
        will be passed to theano's convolution operator resulting in
        possibly faster execution.
    Notes
    -----
    The passed convolutional operators should be 'lazy' constructed, that
    is, without specifying the batch_size, num_channels and image_size. The
    main feature of :class:`ConvolutionalSequence` is that it will set the
    input dimensions of a layer to the output dimensions of the previous
    layer by the :meth:`~.Brick.push_allocation_config` method.
    """
    @lazy(allocation=['num_channels'])
    def __init__(self, layers, num_channels, batch_size=None, image_size=None,
                 border_mode='valid', dropout_layers=[], drop_p=0.5, srng=None, tied_biases=False, **kwargs):
        super(ConvolutionalSequence, self).__init__(**kwargs)
        self.layers = layers
        self.image_size = image_size
        self.num_channels = num_channels
        self.batch_size = batch_size
        self.border_mode = border_mode
        self.tied_biases = tied_biases
        self.dropout_layers = dropout_layers
        self.drop_p = drop_p
        self.srng = srng
        self.children.extend(self.layers) 
        for x in self.layers:
            self.children.extend(x.children)
        print self.children

    def get_dim(self, name):
        if name == 'input_':
            return ((self.num_channels,) + self.image_size)
        if name == 'output':
            return self.layers[-1].get_dim(name)
        return super(ConvolutionalSequence, self).get_dim(name)

    def _push_allocation_config(self):
        num_channels = self.num_channels
        image_size = self.image_size
        for layer in self.layers:
            for attr in ['border_mode', 'tied_biases']:
                setattr(layer, attr, getattr(self, attr))
            if isinstance(layer, MaxPooling):
                layer.input_dim = image_size
            else:
                layer.image_size = image_size
            layer.num_channels = num_channels
            layer.batch_size = self.batch_size

            # Push input dimensions to children
            layer._push_allocation_config()

            # Retrieve output dimensions
            # and set it for next layer
            if isinstance(layer, MaxPooling) and layer.input_dim is not None:
                image_size = layer.get_dim('output')
            if isinstance(layer, ConvolutionalActivation) and layer.image_size is not None:
                output_shape = layer.get_dim('output')
                image_size = output_shape[1:]
                num_channels = output_shape[0]
            
    @application(inputs=['input_'], ouputs=['output'])        
    def training(self, input_):
        out = input_
        for x in self.layers:
            out = x.apply(out)
            if x.name in self.dropout_layers:
                out *= self.srng.binomial(out.shape, p=1.0-self.drop_p, dtype=theano.config.floatX)
        return out
        
    @application(inputs=['input_'], ouputs=['output'])    
    def inference(self, input_):
        out = input_
        for x in self.layers:
            if isinstance(x, ConvolutionalActivation):
                out = x.inference(out)
            else:
                out = x.apply(out)
                if x.name in self.dropout_layers:
                    out = out * (1-self.drop_p)
        return out

class ConvolutionalActivation(Initializable):
    """A convolution followed by an activation function.
    Parameters
    ----------
    activation : :class:`.BoundApplication`
        The application method to apply after convolution (i.e.
        the nonlinear activation function)
    See Also
    --------
    :class:`Convolutional` : For the documentation of other parameters.
    """
    @lazy(allocation=['filter_size', 'num_filters', 'num_channels'])
    def __init__(self, activation, filter_size, num_filters, num_channels,
                 batch_size=None, image_size=None, step=(1, 1),
                 border_mode='valid', tied_biases=False, **kwargs):
        self.convolution = Convolutional(name='conv'+ kwargs['name'])
        self.bn = BatchNorm(name='bn'+ kwargs['name'])
        self.activation = activation
        self.filter_size = filter_size
        self.num_filters = num_filters
        self.num_channels = num_channels
        self.batch_size = batch_size
        self.image_size = image_size
        self.step = step
        self.border_mode = border_mode
        self.tied_biases = tied_biases
        super(ConvolutionalActivation, self).__init__(**kwargs)
        self.children = [self.convolution, self.bn, self.activation]
        

    def _push_allocation_config(self):
        for attr in ['filter_size', 'num_filters', 'step', 'border_mode',
                     'batch_size', 'num_channels', 'image_size',
                     'tied_biases']:
            setattr(self.convolution, attr, getattr(self, attr))
        setattr(self.bn, 'input_dim', self.num_filters)
        
    def get_dim(self, name):
        # TODO The name of the activation output doesn't need to be `output`
        return self.convolution.get_dim(name)
        
    def apply(self, input_):
        out = self.convolution.apply(input_)
        out = self.bn.apply(out)
        out = self.activation.apply(out)
        return out
        
    def inference(self, input_):
        out = self.convolution.apply(input_)
        out = self.bn.inference(out)
        out = self.activation.apply(out)
        return out
