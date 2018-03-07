import pandas as pd
import numpy as np

from sklearn import preprocessing
from numpy import genfromtxt

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
def mnist_to_nparray(path, scale=True):
    [train, y, test] = mnist_to_pdseries(path, scale)
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