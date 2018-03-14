import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.optimizers import Adam
from keras.utils import np_utils

def _base_model(input_shape, nr_of_classes):
    model = Sequential()
    model.add(Dense(64, input_shape=input_shape, activation='relu'))
    model.add(Dense(nr_of_classes, activation='softmax'))
    return model

def _conv_model(input_shape, nr_of_classes):
    model = Sequential()
    print(input_shape)
    input_shape = (1, 28, 28)
    model.add(Conv2D(32, (3, 3), input_shape=input_shape, activation='relu', data_format='channels_first'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    # model.add(Conv2D(64, (3, 3), activation='relu'))
    # model.add(MaxPooling2D(pool_size=(2, 2)))
    # model.add(Conv2D(128, (3, 3), activation='relu'))
    # model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dense(nr_of_classes, activation='softmax'))
    return model

def simple_nn(X_train, y_train, X_test):
    epochs = 5
    batch_size = 64
    learning_rate = 0.01
    optimizer = Adam(lr=learning_rate)

    # reshape data
    y_train_one_hot = np_utils.to_categorical(y_train)
    nr_of_classes = y_train_one_hot.shape[1]

    # make a model
    model = _base_model((X_train.shape[1],), nr_of_classes)
    model.compile(optimizer=optimizer,
                    loss='categorical_crossentropy',
                    metrics=['accuracy']
    )

    # analyze data
    model.fit(X_train, y_train_one_hot, epochs=epochs, batch_size=batch_size)
    simple_nn_predictions = model.predict(X_test)
    simple_nn_predictions = [np.argmax(sample_scores) for sample_scores in simple_nn_predictions]
    return simple_nn_predictions

def convolution_nn(X_train, y_train, X_test):
    epochs = 32
    # epochs = 3
    batch_size = 32
    learning_rate = 0.01
    optimizer = Adam(lr=learning_rate)

    # reshape data
    y_train_one_hot = np_utils.to_categorical(y_train)
    nr_of_classes = y_train_one_hot.shape[1]
    X_train = X_train.reshape(X_train.shape[0], 1, 28, 28).astype('float32')
    X_test = X_test.reshape(X_test.shape[0], 1, 28, 28).astype('float32')

    # make a model
    model = _conv_model((X_train.shape[1],), nr_of_classes)
    print('jooo')
    model.compile(optimizer=optimizer,
                    loss='categorical_crossentropy',
                    metrics=['accuracy']
    )
    print('compileeed')

    # analyze data
    model.fit(X_train, y_train_one_hot, epochs=epochs, batch_size=batch_size)
    print('and fitted')
    predictions = model.predict(X_test)
    print('even predicted')
    predictions = [np.argmax(sample_scores) for sample_scores in predictions]
    return predictions
