import os

import pandas as pd
import numpy as np

from sklearn import preprocessing
from keras.utils import np_utils
### Group of functions related to doing file operations involving the dataset. ###

# Takes a path to a directory and opens returns the train and test set in that directory as a pandas dataframe.
def mnist_to_pdseries(path):
    train = pd.read_csv(path + 'train.csv')
    y = train.ix[:,0]
    train = train.drop('label', 1)
    test = pd.read_csv(path + 'test.csv')
    return [train, y, test]

# Takes a path to a directory and opens returns the train and test set in that directory as a numpy array.
def mnist_to_nparray(path):
    [train, y, test] = mnist_to_pdseries(path)
    train = train.values
    y = y.values
    test = test.values
    return [train, y, test]
    
# Takes list of 28000 predictions and writes them to csv file ready for submission to Kaggle.
def write_results(res, path='', name=''):
    results = pd.DataFrame({'ImageId':range(1,28001), 'Label':res})
    if name == '':
        name = 'results'
    results.to_csv(path + name + '.csv', index=False)
    
# Takes list of categorical values and converts it to Kaggle compatible list.
def categorical_to_class(cat):
    classes = []
    for c in cat:
        classes.append(np.argmax(c))
    return classes

def write_dataset_to_disk(dataset, target, path='', test_set=None, val_set=None):
    """
        Write a dataset to the harddisk.
        Pandas can create dataframe from numpy arrays (which is the format from preprocessing)
    """
    train_frame = pd.DataFrame(data=reshape_matrix_to2d(dataset), dtype='float32')
    train_frame.to_csv(path+'train.csv', index=False, header=False)
    train_target_frame = pd.DataFrame(data=target)
    train_target_frame.to_csv(path+'train_target.csv', index=False, header=False)

    if test_set is not None:
        test_frame = pd.DataFrame(data=reshape_matrix_to2d(test_set))
        test_frame.to_csv(path+'test.csv', index=False, header=False)

    if val_set is not None:
        val_frame = pd.DataFrame(data=reshape_matrix_to2d(val_set[0]))
        val_frame.to_csv(path+'val.csv', index=False, header=False)
        val_target_frame = pd.DataFrame(data=val_set[1])
        val_target_frame.to_csv(path+'val_target.csv', index=False, header=False)

def read_preprocessed_dataset(path):
    """
        Read one of the preprocessed datasets.
        Since reshaping is lost when storing arrays, it is restored here, but only for matrices.
        Target vectors can be stored as 2d matrices - onehot - so they remain unchanged.
    """
    train = reshape_matrix_to3d(pd.read_csv(path+'train.csv', header=None).values)
    target = pd.read_csv(path+'train_target.csv', header=None).values # is already reshaped
    test = reshape_matrix_to3d(pd.read_csv(path + 'test.csv', header=None).values)
    if 'val.csv' in os.listdir(path) and 'val_target.csv' in os.listdir(path):
        val = reshape_matrix_to3d(pd.read_csv(path+'val.csv', header=None).values)
        val_target = pd.read_csv(path+'val_target.csv', header=None).values
        return train, target, test, val, val_target
    return train, target, test

def reshape_matrix_to3d(m):
    """
        Reshape the data to a format that can be used by cnn's, i.e.
        matrices to greyscale (so depth of 1) 2d images of 28 by 28 pixels.
    """
    return m.reshape(m.shape[0], 1, 28, 28).astype('float32')

def reshape_matrix_to2d(m):
    """Reshape 3d images to 2d arrays."""
    num_pixels = m.shape[2] * m.shape[3]
    return m.reshape(m.shape[0], num_pixels).astype('float32')

def reshape_target_to1hot(v):
    """
        Reshape the data to a format that can be used by cnn's, i.e.
        one-hot encoding for the target vectors.
    """
    return np_utils.to_categorical(v)
