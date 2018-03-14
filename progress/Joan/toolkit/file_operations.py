import pandas as pd
import numpy as np
from keras.utils import np_utils

from sklearn import preprocessing

### Group of functions related to doing file operations involving the dataset. ###

# Takes a path to a directory and opens returns the train and test set in that directory as a pandas dataframe.
def mnist_to_pdseries(path, scale=True):
    train = pd.read_csv(path + 'train.csv')
    y = train.ix[:,0]
    train = train.drop('label',1)
    test = pd.read_csv(path + 'test.csv')
    if scale:
        scaler = preprocessing.StandardScaler()
        train = scaler.fit_transform(train)
        test = scaler.transform(test)
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

def reshape_data_for_nn(train, y, test):
    # reshape to be [samples][pixels][width][height]
    train = train.reshape(train.shape[0], 1, 28, 28).astype('float32')
    test = test.reshape(test.shape[0], 1, 28, 28).astype('float32')
    # one hot encode outputs
    y = np_utils.to_categorical(y_train)
