from toolkit.noise import *
from toolkit.file_operations import *

import pandas as pd

from time import time


start = time()
[train,y,test] = mnist_to_nparray('data/', 'train', 'test', False)
[train1,y1,test1] = mnist_to_nparray('data/', 'train', 'test', False)
[train2,y2,test2] = mnist_to_nparray('data/', 'train', 'test', False)
[train3,y3,test3] = mnist_to_nparray('data/', 'train', 'test', False)
print('Data read in ' + str(time() - start) + ' seconds.')

start = time()
add_noisy_pixels(train1, .1)
add_noisy_pixels(train2, .25)
add_noisy_pixels(train3, .5)
print('Train noise added in ' + str(time() - start) + ' seconds.')

start = time()
flip_class_labels(y1, .1)
flip_class_labels(y2, .5)
flip_class_labels(y3, .9)
y = y[:,None]
y1 = y1[:,None]
y2 = y2[:,None]
y3 = y3[:,None]
print('Label noise added in ' + str(time() - start) + ' seconds.')

start = time()
add_noisy_pixels(test1, .1)
add_noisy_pixels(test2, .25)
add_noisy_pixels(test3, .5)
print('Test noise added in ' + str(time() - start) + ' seconds.')


start = time()
write_data(np.append(y, train1, axis=1), 'train10', True)
write_data(np.append(y, train2, axis=1), 'train25', True)
write_data(np.append(y, train3, axis=1), 'train50', True)
write_data(np.append(y1, train, axis=1), 'labels10', True)
write_data(np.append(y2, train, axis=1), 'labels50', True)
write_data(np.append(y3, train, axis=1), 'labels90', True)
write_data(test1, 'test10', False)
write_data(test2, 'test25', False)
write_data(test3, 'test50', False)
print('Written data in ' + str(time() - start) + ' seconds.')


# start = time()
# [train,y,test] = mnist_to_pdseries('data/', 'train', 'test', False)
# [train1,y1,test1] = mnist_to_pdseries('data/', 'train', 'test', False)
# [train2,y2,test2] = mnist_to_pdseries('data/', 'train', 'test', False)
# [train3,y3,test3] = mnist_to_pdseries('data/', 'train', 'test', False)
# print('Data read in ' + str(time() - start) + ' seconds.')

# start = time()
# add_noisy_pixels(train1, .1)
# add_noisy_pixels(train2, .25)
# add_noisy_pixels(train3, .5)
# print('Train noise added in ' + str(time() - start) + ' seconds.')

# start = time()
# flip_class_labels(y1, .1)
# flip_class_labels(y2, .5)
# flip_class_labels(y3, .9)
# print('Label noise added in ' + str(time() - start) + ' seconds.')

# start = time()
# add_noisy_pixels(test1, .1)
# add_noisy_pixels(test2, .25)
# add_noisy_pixels(test3, .5)
# print('Test noise added in ' + str(time() - start) + ' seconds.')


# start = time()
# write_data(train1.insert(0, 'label', y), 'train10')
# write_data(train2.insert(0, 'label', y), 'train25')
# write_data(train3.insert(0, 'label', y), 'train50')
# write_data(train.insert(0, 'label', y1), 'labels10')
# write_data(train.insert(0, 'label', y2), 'labels50')
# write_data(train.insert(0, 'label', y3), 'labels90')
# write_data(test1, 'test10')
# write_data(test2, 'test25')
# write_data(test3, 'test50')
# print('Written data in ' + str(time() - start) + ' seconds.')