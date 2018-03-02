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

import sys
# Add the folder path to the sys.path list
sys.path.append('/progress/Joan/toolkit/')
from file_operations import mnist_to_pdseries, write_results

DATA_FOLDER = 'progress/Joan/data/'
RESULTS_FOLDER = 'progress/Michel/results/'

def classify(X_train, y_train, X_test, classifiers):
    nr_of_samples = X_train.shape[0]

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

def main():
    """Run analyses"""
    [train, y, test] = mnist_to_pdseries(DATA_FOLDER)
    # classify(train, y, test, ['svm', 'knn'])

if __name__ == "__main__":
    main()
