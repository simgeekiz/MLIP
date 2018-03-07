from toolkit.file_operations import mnist_to_nparray, write_results, categorical_to_class
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils

import toolkit.noise as noise

print('Reading and reshaping data')
[X_train, y_train, X_test] = mnist_to_nparray('data/', False)
noise.add_noisy_pixels(X_train, 0.25)

X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)
X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)
y_train = np_utils.to_categorical(y_train, 10)


print('Constructing model')
model = Sequential()

model.add(Convolution2D(32, 3, 3, activation='relu', input_shape=(28,28,1)))
model.add(Convolution2D(32, 3, 3, activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam')

print('Fitting model')
model.fit(X_train, y_train, epochs=25, batch_size=32)

print('Making predictions')
predictions = model.predict(X_test)

output_name = input('Enter name for outputfile\n')
print('Writing results to .csv file')
write_results(categorical_to_class(predictions), 'results/', output_name)
