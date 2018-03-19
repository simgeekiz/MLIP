from toolkit.testing import *
from toolkit.file_operations import mnist_to_nparray
from keras.utils import np_utils
from time import time

def reshape_data(x, y):
	x = x.reshape(x.shape[0], 28, 28, 1)
	y = np_utils.to_categorical(y, 10)
	return x, y


print('Reading data...')
[x,y,_] = mnist_to_nparray('data/', 'train', 'test', False)
[x_10,_,_] = mnist_to_nparray('data/', 'train10', 'test', False)
[x_25,_,_] = mnist_to_nparray('data/', 'train25', 'test', False)
[x_50,_,_] = mnist_to_nparray('data/', 'train50', 'test', False)
[_,y10,_] = mnist_to_nparray('data/', 'labels10', 'test', False)
[_,y50,_] = mnist_to_nparray('data/', 'labels50', 'test', False)
[_,y90,_] = mnist_to_nparray('data/', 'labels90', 'test', False)
x, y = reshape_data(x, y)
x_10, y10 = reshape_data(x_10, y10)
x_25, y50 = reshape_data(x_25, y50)
x_50, y90 = reshape_data(x_50, y90)


print('Training models...')
n = int(.9*42000)
model = train_conv_net(x[:n,:],y[:n],3)
lab10 = train_conv_net(x[:n,:],y10[:n],3)
lab50 = train_conv_net(x[:n,:],y50[:n],3)
lab90 = train_conv_net(x[:n,:],y90[:n],3)
tra10 = train_conv_net(x_10[:n,:],y[:n],3)
tra25 = train_conv_net(x_25[:n,:],y[:n],3)
tra50 = train_conv_net(x_50[:n,:],y[:n],3)


print('Making predictions...')
pred0 = categorical_to_class(model.predict(x[n:,:]))
pred1 = categorical_to_class(tra10.predict(x[n:,:]))
pred2 = categorical_to_class(tra25.predict(x[n:,:]))
pred3 = categorical_to_class(tra50.predict(x[n:,:]))
pred4 = categorical_to_class(model.predict(x_10[n:,:]))
pred5 = categorical_to_class(model.predict(x_25[n:,:]))
pred6 = categorical_to_class(model.predict(x_50[n:,:]))
pred7 = categorical_to_class(lab10.predict(x[n:,:]))
pred8 = categorical_to_class(lab50.predict(x[n:,:]))
pred9 = categorical_to_class(lab90.predict(x[n:,:]))

y = categorical_to_class(y[n:])


print('RESULTS')
accuracy(pred0, y, 'Normal')
accuracy(pred1, y, 'Tra10')
accuracy(pred2, y, 'Tra25')
accuracy(pred3, y, 'Tra50')
accuracy(pred4, y, 'Tst10')
accuracy(pred5, y, 'Tst25')
accuracy(pred6, y, 'Tst50')
accuracy(pred7, y, 'Lab10')
accuracy(pred8, y, 'Lab50')
accuracy(pred9, y, 'Lab90')