import numpy 

from transformers import ZCA, ContrastNorm, Whitening
from fuel.transformers import ForceFloatX
from fuel.datasets import CIFAR10
from fuel.streams import DataStream
from fuel.schemes import ShuffledScheme, SequentialScheme

train_dataset = CIFAR10(['train'], sources=('features', 'targets'))
test_dataset = CIFAR10(['test'], sources=('features', 'targets'))

batch_size = 400 #Batch size for training
batch_size_mon = 4000 # Batch size for monitoring and batch normalization
n_batches = int(numpy.ceil(float(train_dataset.num_examples)/batch_size_mon))
num_protos = 10

train_data = train_dataset.get_data(request=range(train_dataset.num_examples))[0]
print train_data.shape
#cnorm = ContrastNorm()
#train_data = cnorm.apply(train_data)
#whiten = ZCA()
#whiten.fit(3072, train_data)

def preprocessing(data_stream, num_examples, batch_size):
    return data_stream
    #return ForceFloatX(Whitening(data_stream, ShuffledScheme(num_examples, batch_size), whiten, cnorm), which_sources=('features',))

train_stream_mon = preprocessing(DataStream(train_dataset), train_dataset.num_examples, train_dataset.num_examples)
test_stream = preprocessing(DataStream(test_dataset), test_dataset.num_examples, test_dataset.num_examples)

trainset = train_dataset.get_data(request=range(train_dataset.num_examples))
testset = test_dataset.get_data(request=range(test_dataset.num_examples))

print trainset[0].shape
print trainset[1].shape
print testset[0].shape 

import cPickle

cPickle.dump({'train_x': trainset[0], 'train_y': trainset[1], 'test_x': testset[0], 'test_y': testset[1]}, open('/Tmp/devries/cifar10_raw.pkl', 'wb'))