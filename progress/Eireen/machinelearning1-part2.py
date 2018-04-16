
# coding: utf-8

# ## Machine Learning in Practice trials

# In[1]:


import pandas as pd
import numpy as np

from numpy import genfromtxt

# Takes a path to a directory and opens returns the train and test set in that directory as a pandas dataframe.
def mnist_to_csv(path):
	train = pd.read_csv(path + 'train.csv')
	y = train.ix[:,0].values
	train = train.drop('label',1).values
	test = pd.read_csv(path + 'test.csv').values
	return [train, y, test]

# Takes a path to a directory and opens returns the train and test set in that directory as a numpy array.
def mnist_to_nparray(path):
	[train, y, test] = mnist_to_csv(path)
	train = train.values
	y = y.values
	test = test.values
	return [train, y, test]
	
# Takes list of 28000 predictions and writes them to csv file ready for submission to Kaggle.
def write_results(res, path=''):
	results = pd.DataFrame({'ImageId':range(1,28001), 'Label':res})
	results.to_csv(path + 'results.csv', index=False)
	
# Takes list of categorical values and converts it to Kaggle compatible list.
def categorical_to_class(cat):
	classes = []
	for c in cat:
		classes.append(np.argmax(c))
	return classes
	



# In[2]:


trainxp, trainyp, testset = mnist_to_csv("data\\")

separationIndex = int( len(trainxp)*0.1)
trainx = trainxp[separationIndex:len(trainxp)]
testx = trainxp[0:separationIndex-1]
trainy = trainyp[separationIndex:len(trainxp)]
testy = trainyp[0:separationIndex-1]



# In[4]:


#temporary noise toolkit
import copy
from random import random, seed, randint
def add_noisy_pixels(x, p, rseed=-1):
	newx = copy.copy(x)
	if rseed != -1:
		seed(rseed)
	for i in range(0,len(x)):
		for j in range(0, len(x[0])):
			if random() < p:
				newx[i][j] = randint(0,255)
	return newx


# In[5]:


#import toolkit.noise
#import random
testx_noisy_pixels = add_noisy_pixels(testx, 0.6, 1)
trainx_noisy_pixels = add_noisy_pixels(trainx, 0.6, 2)



# In[6]:


print(testset.shape)
testset_sq = np.reshape(testset,(len(testset),28,28,1))

testset_noisy_sq = np.reshape(add_noisy_pixels(testset, 0.6, 3),(len(testset),28,28,1))


# In[7]:


trainx_sq = np.reshape(trainx,(len(trainx), 28, 28,1))
testx_sq = np.reshape(testx,(len(testx),28,28,1))
testx_sq_noisy = np.reshape(testx_noisy_pixels,(len(testx),28,28,1))
trainx_sq_noisy = np.reshape(trainx_noisy_pixels,(len(trainx),28,28,1))


# In[8]:


import matplotlib
import matplotlib.pyplot as plot
#plot test

for i in range(0,3):
    print(testy[i])
    plot.imshow(np.reshape(testx_sq_noisy, (len(testx), 28,28)) [i], cmap='gray')
    plot.show()


# In[9]:


from skimage.filters import gabor_kernel
from skimage import io
from matplotlib import pyplot as pltk
import scipy

def extract_features(img): 
    imgsq = np.reshape(img,(28,28))
    imgblurred = scipy.ndimage.filters.gaussian_filter(imgsq, 1.3)
    features = [imgblurred]
    #gabor filter
   # thetas = [0, 0.3, 0.6, 0.9, 1.2, 1.6, -0.3, -0.6]
    thetas = [0, 0.6, 1.6, -0.6]
    
    for t in thetas:
        kernel = np.real(gabor_kernel(0.4, t))
        conv = scipy.signal.fftconvolve(imgblurred, kernel, mode='same')
        
        features.append(conv)
        
    #features.append(imgblurred)
    return features

        #normalisation idea: extract lines, sort by orientation, save length, normalize length to height of shape/ longest line
    
    #gabor filter wil react on lines that are not necessarily matching in direction. Maybe compute all gabors, 
    #sort by strongest reaction
    
    #use (inverse) gaussian for circle detection??????



# In[10]:


test = extract_features(testx_noisy_pixels[2])

for each in test:
    plot.imshow(each, cmap='gray')
    plot.show()


# In[ ]:


feature_train = [extract_features(x) for x in trainx_noisy_pixels]
feature_test = [extract_features(x) for x in testx_noisy_pixels]


# In[ ]:


#feature_train_mini = [extract_features(x) for x in trainx_noisy_pixels]
#feature_test_mini = [extract_features(x) for x in testx_noisy_pixels]


# ### Neural Network

# In[ ]:


import tensorflow as tf
import keras
#feature_train = [extract_features(x) for x in trainx_noisy_pixels]
#feature_test = [extract_features(x) for x in testx_noisy_pixels]


from keras.models import Sequential
from keras.layers import Dense, Activation, BatchNormalization, Conv2D, MaxPooling2D, Dropout, Flatten



# In[16]:



def build_neural_network(data_size, n_classes):
    #convolution
    network = Sequential()
    network.add(Conv2D(20, (5,5), input_shape = data_size))
    network.add(BatchNormalization())
    network.add(Activation('relu'))
    
    network.add(Dropout(0.25))
    
    network.add(Conv2D(20, (5,5), input_shape = data_size))
    network.add(BatchNormalization())
    network.add(Activation('relu'))
    
    network.add(Dropout(0.25))
    
    network.add(Flatten())
    
    network.add(Dense(20))
    network.add(BatchNormalization())
    network.add(Activation('relu'))
    
    
    
    
    network.add(Dense(n_classes, activation='softmax'))
    
    
    
    return network





# In[17]:


nnetwork=build_neural_network((28,28,1), 10)
# build neural network object
#network=build_neural_network([5*28*28], 10)
#network=build_neural_network((37800, 5, 28, 28), 10)


# In[18]:


from keras import optimizers

loss = "categorical_crossentropy" 
sgd = optimizers.SGD(lr=0.7) 
metrics = ['accuracy'] 

nnetwork.compile(loss=loss, optimizer=sgd, metrics=metrics)
nnetwork.summary()


# In[19]:


trainy_vector = keras.utils.to_categorical(trainy, num_classes = 10)
testy_vector = keras.utils.to_categorical(testy, num_classes = 10)



# In[20]:


#scores=network.evaluate(np.reshape(np.array(feature_train), (len(feature_train), 5*28*28)), trainy_vector, batch_size=1)
scores= nnetwork.evaluate(np.array(trainx_sq_noisy), trainy_vector, batch_size=128)
val_loss=scores[0]
val_acc=scores[1]
print(val_loss)
print(val_acc)


# In[21]:


print(trainx_sq_noisy.shape)
trainy_vector.shape
testx_sq_noisy.shape


# In[23]:


#training

epochs = 10


for i in range(0, epochs):
    print(10+i)
    results = nnetwork.fit(np.array(trainx_sq_noisy), trainy_vector, batch_size=30)
    print("train accuracy: ")
    print(results.history['acc'])
    scores = nnetwork.evaluate(np.array(testx_sq_noisy), testy_vector)
    print('test accuracy:z ')
    print(scores[1])
                              
    



# In[ ]:



#thanks joan

def write_results(res, path='', name=''):
    results = pd.DataFrame({'ImageId':range(1,28001), 'Label':res})
    if name == '':
        name = 'results'
    results.to_csv(path + name + '.csv', index=False)
    
predictions = nnetwork.predict(testset_noisy_sq)

output_name = input('Enter name for outputfile\n')
print('Writing results to .csv file')
write_results(categorical_to_class(predictions), 'results/', output_name)


