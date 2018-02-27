### Group of functions related to adding noise over the dataset. ###from random import random, randint, seed

import pandas as pd
import numpy as np

# Changes class label to a random other class with probability p.
def flip_class_labels(y, p, seed=-1):
	if seed != -1:
		seed(seed)
	
	for i, label in enumerate(y):
		if random() < p:
			y[i] = (label + randint(1,9)) % 10

# For every pixel in the given dataset change the value to a random grayscale value with probability p.
def add_noisy_pixels(x, p, seed=-1):
    if seed != -1:
        seed(seed)
    
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
    num = p * x.size
    
    if isinstance(x, np.ndarray):
        red = x[:int(num/784),:]
    elif isinstance(x, pd.DataFrame) 
        red = x.head(n=int(num/784))
    elif isinstance(x, pd.Series): 
        red = x.head(n=int(num))
    else:
        raise ValueError('Unsupported type in toolkit.noise -> reduce_dataset')
        
    return red