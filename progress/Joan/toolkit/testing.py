
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, BatchNormalization
from keras.layers import Convolution2D, MaxPooling2D, Activation

import numpy as np

def train_conv_net(train, y, epoch_num=25):	
	model = Sequential()
	
	model.add(Convolution2D(32, 3, 3, input_shape=(28,28,1)))
	model.add(BatchNormalization())
	model.add(Activation('relu'))
	model.add(Convolution2D(32, 3, 3))
	model.add(BatchNormalization())
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2,2)))
	model.add(Dropout(0.25))
	
	model.add(Flatten())
	model.add(Dense(128))
	model.add(Activation('relu'))
	model.add(BatchNormalization())
	model.add(Dropout(0.5))
	model.add(Dense(10))
	model.add(BatchNormalization())
	model.add(Activation('softmax'))
	
	model.compile(loss='categorical_crossentropy', optimizer='adam')
	
	model.fit(train, y, epochs=epoch_num, batch_size=32)
	return model
	
# Computes accuracy of predictions given the set of real labels.
def accuracy(predictions, labels, name=''):
	cor = 0
	tot = 0
	
	for pred, lab in zip(predictions, labels):
		tot += 1
		if pred == lab:
			cor += 1
	
	print(name + ' accuracy:\t' + str(cor/tot))

# Takes list of categorical values and converts it to Kaggle compatible list.
def categorical_to_class(cat):
    classes = []
    for c in cat:
        classes.append(np.argmax(c))
    return classes