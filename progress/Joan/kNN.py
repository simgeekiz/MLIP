from sklearn.neighbors import NearestNeighbors
from toolkit.file_operations import mnist_to_pdseries, write_results

import pandas as pd

def indicesToNum(indices, labels):
	nums = []
	for i in indices:
		nums.append(labels[i])
	return max(set(nums), key=nums.count)


print('Reading data')
[train, y, test] = mnist_to_pdseries('data/')
	

print('Fitting model')
K = 3
nn = NearestNeighbors(K)
nn.fit(train,y)


print('Running test')
from time import time
a = time()

(_, indices) = nn.kneighbors(test)

res = []
for row in indices:
	res.append(indicesToNum(row,y))
b = time()
print('Results computed in ' + str((b-a)/60) + ' minutes.')

write_results(res, 'results/', 'nearest_neighbor_results')
print('Results written to .csv file')
