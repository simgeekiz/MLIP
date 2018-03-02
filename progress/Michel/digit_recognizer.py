"""
    Module to recognize digits from the MNIST database.
    Challenge derived from Kaggle: https://www.kaggle.com/c/digit-recognizer
"""
import pandas as pd
import numpy as np
import math

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn import svm

from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.optimizers import Adam
from keras.utils import np_utils

from file_operations import mnist_to_pdseries, write_results

DATA_FOLDER = 'progress/Joan/data/'
RESULTS_FOLDER = 'progress/Michel/results/'

def classify(X_train, y_train, X_test, classifiers):
    nr_of_samples = X_train.shape[0]
    classes = np.unique(y_train)
    nr_of_classes = len(classes)

    # svm implementation
    if 'svm' in classifiers:
        svm_clf = svm.SVC().fit(X_train, y_train)
        svm_predictions = svm_clf.predict(X_test)
        write_results(svm_predictions, RESULTS_FOLDER, 'svm_results')

    # k-nearest neighbors
    if 'knn' in classifiers:
        # rule of thumb to take k as the square root of the number of training samples
        k = math.floor(math.sqrt(nr_of_samples))
        kneigh_clf = KNeighborsClassifier(n_neighbors=k).fit(X_train, y_train)
        k_predictions = kneigh_clf = kneigh_clf.predict(X_test)
        write_results(k_predictions, RESULTS_FOLDER, 'k_results')
    
    if 'simple_nn' in classifiers:
        epochs = 32
        batch_size = 64
        learning_rate = 0.01
        optimizer = Adam(lr=learning_rate)

        model = Sequential()
        model.add(Dense(64, input_shape=(X_train.shape[1],), activation='relu'))
        model.add(Dense(nr_of_classes, activation='softmax'))
        model.compile(optimizer=optimizer,
                        loss='categorical_crossentropy',
                        metrics=['accuracy']
        )

        y_train_one_hot = np_utils.to_categorical(y_train, num_classes=nr_of_classes)
        model.fit(X_train, y_train_one_hot, epochs=epochs, batch_size=batch_size)
        simple_nn_predictions = model.predict(X_test)
        simple_nn_predictions = [np.argmax(sample_scores) for sample_scores in simple_nn_predictions]
        write_results(simple_nn_predictions, RESULTS_FOLDER, 'simple_nn_results_3')

def main():
    """Run analyses"""
    [train, y, test] = mnist_to_pdseries(DATA_FOLDER)
    # the following algorithms can be added to 'algorithms'.
    # svm - Support Vector Machine
    # knn - K-nearest neighbors
    # simple_nn - Simple Neural Network
    algorithms = ['simple_nn']
    classify(train, y, test, algorithms)

if __name__ == "__main__":
    main()
