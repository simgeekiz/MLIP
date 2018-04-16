"""
    Module to preprocess the MNIST data set.
    Operations include:
        - scaling or normalization
        - reshaping (so that it can be handled by cnn's)
        - data augmentation:
            This is done in two ways:
                1 the total dataset is augmented - method name: create_full
                2 only the trainingset is augmented - method name: create_splitted
"""
import pandas as pd
import numpy as np
import os
from imgaug import augmenters as iaa

from sklearn import preprocessing
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

from toolkit import mnist_to_pdseries, write_results, write_dataset_to_disk, reshape_matrix_to3d,  reshape_target_to1hot

DATA_FOLDER = 'Joan/data/'
RESULTS_FOLDER = 'Michel/preprocessed_datasets/'

class Augmentor(object):
    """Class that augments data."""
    def __init__(self, data, target):
        self.data = data
        self.target = target

    @staticmethod
    def __blur(data, sigma=(0, 1.0)):
        """Blur images with a certain sigma."""
        seq_blur = iaa.Sequential([iaa.GaussianBlur(sigma=sigma)])
        return seq_blur.augment_images(data) 
    
    @staticmethod
    def __shuffle(data):
        """
            The following steps are performed:
            1. scale images to 80-120% of their size, individually per axis
            2. translate by -20 to +20 percent (per axis)
        """
        seq_shuffle = iaa.Sequential([iaa.Affine(
            scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
            translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)}
        )])
        return seq_shuffle.augment_images(data)

    @staticmethod
    def __crop(data):
        """Crop the images."""
        seq_crop = iaa.Sequential([iaa.Crop()])
        return seq_crop.augment_images(data)
    
    def augment(self, operations=None, should_shuffle=True):
        """Augment data on the operations, by default: all available methods in this class."""
        if operations is None:
            operations = [self.__blur, self.__shuffle, self.__crop]
        dataset = self.data
        target = self.target
        for operation in operations:
            dataset = np.concatenate([dataset, operation(self.data)])
            target = np.concatenate([target, self.target])
        if should_shuffle:
            dataset, target = shuffle(dataset, target)
        return dataset, target

class PreProcessor(object):
    """Class that preprocesses the data"""
    def __init__(self, path):
        # read data
        X_train, y_train, X_test = mnist_to_pdseries(path)

        # scale the data when initializing directly.
        scaler = preprocessing.StandardScaler()
        self.X_train = scaler.fit_transform(X_train)
        self.y_train = y_train

        # reshaping of the test can be done as it is never touched afterwards.
        self.X_test = reshape_matrix_to3d(scaler.transform(X_test))

    def create_full(self):
        """Augment the total training set."""
        # reshape the variables
        X_train = reshape_matrix_to3d(self.X_train)
        y_train = reshape_target_to1hot(self.y_train)
        
        # augment data
        augmentor = Augmentor(X_train, y_train)
        dataset, target = augmentor.augment()
        write_dataset_to_disk(dataset, target, RESULTS_FOLDER+'full/', test_set=self.X_test)

    def create_splitted(self, val_size=0.15):
        """
            Augment the training part of a splitted dataset,
            so that the validation set is not augmented.
        """
        # split the dataset
        X_train_splitted, X_val_splitted, y_train_splitted, y_val_splitted = train_test_split(self.X_train, self.y_train, test_size=val_size)
        
        # reshape every variable
        X_train_splitted = reshape_matrix_to3d(X_train_splitted)
        X_val_splitted = reshape_matrix_to3d(X_val_splitted)
        y_train_splitted = reshape_target_to1hot(y_train_splitted)
        y_val_splitted = reshape_target_to1hot(y_val_splitted)

        # augment data
        augmentor = Augmentor(X_train_splitted, y_train_splitted)
        dataset, target = augmentor.augment()
        write_dataset_to_disk(dataset, target, RESULTS_FOLDER+'splitted/', test_set=self.X_test, val_set=(X_val_splitted, y_val_splitted))

def main():
    """All preprocessing steps."""
    p = PreProcessor(DATA_FOLDER)
    p.create_full()
    p.create_splitted()

if __name__ == "__main__":
    main()
