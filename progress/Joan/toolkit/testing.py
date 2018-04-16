
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, BatchNormalization
from keras.layers import Convolution2D, MaxPooling2D, Activation, Conv2D, MaxPool2D
from keras.utils import np_utils
from keras.optimizers import Adam
from keras.callbacks import ReduceLROnPlateau

import numpy as np

def train_simges_net(train, y, epoch_num=25, model=None, verbose=1, batch=86):
	if model == None:
		optimizer = Adam()
		learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc', patience=3, factor=0.5, min_lr=0.0001)
		
		model = Sequential()

		model.add(Conv2D(filters=32, kernel_size=(5,5), padding='Same', activation='relu', input_shape=(28,28,1)))
		model.add(Conv2D(filters=32, kernel_size=(5,5), padding='Same', activation ='relu'))
		model.add(MaxPool2D(pool_size=(2,2)))
		model.add(Dropout(0.25))

		model.add(Flatten())
		model.add(Dense(128, activation = "relu"))
		model.add(Dropout(0.5))

		model.add(Dense(10, activation = "softmax", name = "Output_Layer"))
		model.compile(loss='categorical_crossentropy', optimizer='adam')
	
	model.fit(train, y, epochs=epoch_num, batch_size=batch, verbose=verbose)
	
	return model


def train_conv_net(train, y, epoch_num=25, model=None, batch=32, verbose=1):

	if model == None:
		optimizer = Adam()
		learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc', patience=3, factor=0.5, min_lr=0.0001)
		
		model = Sequential()
		
		model.add(Convolution2D(32, 4, 4, input_shape=(28,28,1)))
		model.add(BatchNormalization())
		model.add(Activation('relu'))
		
		model.add(Convolution2D(32, 3, 3))
		model.add(BatchNormalization())
		model.add(Activation('relu'))
		
		model.add(MaxPooling2D(pool_size=(2,2)))
		
		model.add(Dropout(0.25))
		
		model.add(Flatten())
		
		model.add(Dense(128))
		model.add(BatchNormalization())
		model.add(Activation('relu'))
		
		model.add(Dropout(0.5))
		
		model.add(Dense(10))
		model.add(Activation('softmax'))
		
		model.compile(loss='categorical_crossentropy', optimizer='adam')
	
	model.fit(train, y, epochs=epoch_num, batch_size=batch, verbose=verbose)
	
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