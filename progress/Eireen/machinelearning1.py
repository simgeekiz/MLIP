
# coding: utf-8

# ## Machine Learning in Practice trials

# In[2]:


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
	



# In[3]:


trainxp, trainyp, testset = mnist_to_csv("data\\")

separationIndex = int( len(trainxp)*0.1)
trainx = trainxp[separationIndex:len(trainxp)]
testx = trainxp[0:separationIndex-1]
trainy = trainyp[separationIndex:len(trainxp)]
testy = trainyp[0:separationIndex-1]


# In[4]:


trainx_sq = np.reshape(trainx,(len(trainx), 28, 28))
testset_sq = np.reshape(testset,(len(testset),28,28))


# In[5]:


import matplotlib
import matplotlib.pyplot as plot
#plot test

for i in range(0,3):
    plot.imshow(trainx_sq[i], cmap='gray')
    plot.show()


# In[12]:


from skimage.filters import gabor_kernel
from skimage import io
from matplotlib import pyplot as pltk
import scipy

def extract_features(img): 
    features = [img]
    #gabor filter
    thetas = [0, 0.6, 1.6, -0.6]
    
    for t in thetas:
        kernel = np.real(gabor_kernel(0.4, t))
        conv = scipy.signal.fftconvolve(img, kernel)#, mode='same')
        
        features.append(conv)
    return features

        #normalisation idea: extract lines, sort by orientation, save length, normalize length to height of shape/ longest line
    
    #gabor filter wil react on lines that are not necessarily matching in direction. Maybe compute all gabors, 
    #sort by strongest reaction
    
    #use (inverse) gaussian for circle detection??????



# In[6]:


test = extract_features(train[3])


for each in test:
    plot.imshow(each, cmap='gray')
    plot.show()


# In[ ]:


#kernel = np.real(gabor_kernel(0.4))
#len(kernel[1])
len(test[2])
len(testset[1])


# ## Random Forest classifier:

# In[7]:


from sklearn.ensemble import RandomForestClassifier

classifier = RandomForestClassifier()
classifier.fit(trainx, trainy)


# In[8]:


#test the testset:
total = len(testx)
classifications = classifier.predict(testx)
right = sum(classifications == testy)
percentage = right / total
print(right)
print(total)
print(percentage)


# In[46]:


#testset_sq = np.reshape(trainx,(len(trainx), 28, 28))

for i in range(0,10):
    p = classifier.predict(testset[i].reshape(1,-1))
    print(p)
   # plot.imshow(testset[i].reshape(( 28,28 )), cmap='gray')
    #plot.show()
    


# In[9]:


trainx.shape


# ## Evolutionary algorithm:

# In[ ]:


def get_score(w):
    score = 0
    for i in range(0,len(trainx)): #(trainx,trainy):
        res = trainx[i].dot(w)
        imax = np.argmax(res)
        if (imax == trainy[i]):
            score +=1
    return score/len(trainx)
    
def get_survivors(surv, scores, weights):
    best_i = list(range(0,surv))
    best_v = scores[0:surv-1]
    for i in range(surv, len(scores)):
        if(scores[i]>min(best_v)):
            min_i = np.argmin(best_v)
            best_v[min_i] = scores[i]
            best_i[min_i] = i
    survivors = []
    for i in best_i:
        survivors.append(weights[i])
    return survivors

def get_next_gen(parents, genSize, mutRate):
    nextGen = []
    while(len(nextGen)<genSize):
        
        p1 = parents[np.random.randint(0,len(parents)-1)] 
        p2 = parents[np.random.randint(0,len(parents)-1)] 
        mutationp = np.random.rand(features,10)
        mask = (np.random.rand(len(p1), len(p1[0])) > 0.5)
        mutMask = (np.random.rand(len(p1), len(p1[0])) < mutRate)
        kid = ((p1*mask) + (p2* (mask == 0)))
        if(len(nextGen) % 2 == 0):
            kid = kid*(mutMask == 0) + mutMask*mutationp
        nextGen.append(kid)
    return nextGen
        

features = len(testx[1])
generationSize = 10
survivorSize = 4
mutationRate = 0.3   #the higher, the more mutations
weights = []
for i in range(0,generationSize):
    weights.append(np.random.rand(features,10))
best_score = 0
best_score_w = []

scores = []
for i in range(0,50):
    print("iteration: ")
    print(i)
    for w in weights:
        scores.append(get_score(w))
    best_score = max(scores)
    best_score_w = np.argmax(scores)
    mutationRate = (1-best_score)/1.4
    print(best_score)
    parents = get_survivors(survivorSize, scores, weights)
    weights = get_next_gen(parents, generationSize, mutationRate)
    scores = []
    





# 
# ## ....../??????
