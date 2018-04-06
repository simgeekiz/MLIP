
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, BatchNormalization
from keras.layers import Convolution2D, MaxPooling2D, Activation
from keras.utils import np_utils

import numpy as np

def train_conv_net(train, y, epoch_num=25, fault_focus=False, model=None):
	if model == None:
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
	
	if not fault_focus:
		model.fit(train, y, epochs=epoch_num, batch_size=32)
	else:
		yclass = categorical_to_class(y)
		for i in range(epoch_num):
			model.fit(train, y, batch_size=32)
			pred = categorical_to_class(model.predict(train))
			(misses, missesy) = mistakes(train, pred, yclass)
			model.fit(misses, np_utils.to_categorical(missesy, 10), verbose=0)
	
	return model


# Return nparray of all images that were incorrectly predicted.
def mistakes(trainset, predictions, labels):
	indices = []
	for i,p in enumerate(predictions):
		if not predictions[i] == labels[i]:
			indices.append(i)
	return (np.take(trainset, indices, axis=0), np.take(labels, indices))

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