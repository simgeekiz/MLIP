### Group of functions related to adding noise over the dataset. ###
from random import random, randint, seed
from matplotlib import pyplot as plt

import pandas as pd
import numpy as np


# Changes class label to a random other class with probability p.
def flip_class_labels(y, p, sd=1):
	seed(sd)
	
	for i, label in enumerate(y):
		if random() < p:
			y[i] = (label + randint(1,9)) % 10
			

# For every pixel in the given dataset change the value to a random grayscale value with probability p. Takes approximately 25-30 minutes for the whole dataset.
def add_noisy_pixels(x, p, sd=1):
	seed(sd)
	
	if isinstance(x, np.ndarray):
		for img in x:
			for i,v in enumerate(img):
				if random() < p:
					img[i] = randint(0,255)
	elif isinstance(x, pd.DataFrame):
		for _,img in x.iterrows():
			for i,_ in enumerate(img):
				if random() < p:
					img.iloc[i] = randint(0,255)
	else:
		raise ValueError('Unsupported type in toolkit.noise -> add_noisy_pixels')

		
# Reduce the given dataset to p percent of the original size.		
def reduce_dataset(x, p):
	num = int(p * x.shape[0])
	
	if isinstance(x, np.ndarray) and len(x.shape) > 1:
		red = x[:num,:]
	elif isinstance(x, np.ndarray) and len(x.shape) == 1:
		red = x[:num]
	elif isinstance(x, pd.DataFrame):
		red = x.head(n=num)
	elif isinstance(x, pd.Series): 
		red = x.head(n=num)
	else:
		raise ValueError('Unsupported type in toolkit.noise -> reduce_dataset')
		
	return red

	
# Plots a given digit.
def display_digit(digit):
	if isinstance(digit, pd.Series):
		digit = digit.values
	
	digit = digit.reshape((28,28))
	plt.imshow(digit, interpolation='nearest')
	plt.show()