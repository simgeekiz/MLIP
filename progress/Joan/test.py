from toolkit.testing import *
from toolkit.file_operations import mnist_to_nparray
from keras.utils import np_utils
from time import time

def reshape_data(x, y):
	x = x.reshape(x.shape[0], 28, 28, 1)
	y = np_utils.to_categorical(y, 10)
	return x, y


print('Reading data...')
[x,y,_] = mnist_to_nparray('data/', 'train10', 'test', False)
[x_reg,z,_] = mnist_to_nparray('data/', 'train', 'test', False)
x, y = reshape_data(x, y)
x_reg, z = reshape_data(x_reg, z)

print('Training models...')
n = int(.9*42000)
model = train_conv_net(x[:n,:],y[:n],3)


print('Making predictions...')
pred1 = categorical_to_class(model.predict(x_reg[n:,:]))
pred2 = categorical_to_class(model.predict(x[n:,:]))
y = categorical_to_class(y[n:])


print('RESULTS')
accuracy(pred1, y, 'Normal')
accuracy(pred2, y, 'Tra10')