
import time
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split

from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import Adam
from keras.callbacks import ReduceLROnPlateau

print("Loading data...")

noisiness = '_noisy' # options; '', '_noisy', 'Joan'

train_path = '../Joan/data/train.csv' if noisiness == 'Joan' else 'data/train{}.csv'.format(noisiness)
train = pd.read_csv(train_path)

test_path = '../Joan/data/test.csv' if noisiness == 'Joan' else 'data/test{}.csv'.format(noisiness)
test = pd.read_csv(test_path)

y_train = train.pop('label')
X_train = train

del train

print("Preprocessing...")

# Normalization
X_train = X_train/255.0
test = test/255.0

# One hot encoding
y_train = to_categorical(y_train, num_classes = 10)

print("Splitting...")

X_train, X_val, y_train, y_val = train_test_split(X_train, y_train,
                                                  test_size=0.1,
                                                  random_state=42)

print("Setting optimizer...")

optimizer = Adam()

learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc',
                                            patience=3,
                                            verbose=1,
                                            factor=0.5,
                                            min_lr=0.0001)

print("Creating the network...")

epochs = 30
batch_size = 86

model = Sequential()

model.add(Dense(300, activation = "relu", input_shape=X_train.shape[1:]))
model.add(Dropout(0.25))

model.add(Dense(100, activation = "relu"))
model.add(Dropout(0.25))

model.add(Dense(100, activation = "relu"))
model.add(Dropout(0.25))

model.add(Dense(100, activation = "relu"))
model.add(Dropout(0.25))

model.add(Dense(200, activation = "relu"))
model.add(Dropout(0.25))

model.add(Dense(10, activation = "softmax", name = "Output_Layer"))

print("\nNetwork summary:")
print(model.summary())

print("Compiling...")

model.compile(loss='categorical_crossentropy',
              optimizer=optimizer,
              metrics=['accuracy'])

print("Fitting...")

start_time = time.time()

model_history = model.fit(X_train, y_train,
                          batch_size=batch_size,
                          epochs=epochs,
                          validation_data=(X_val, y_val),
                          verbose=2,
                          callbacks=[learning_rate_reduction])

print("Fitting took {} minutes".format((time.time() - start_time)/60))

print("Predicting...")

results = model.predict(test)

results = np.argmax(results, axis = 1)
results = pd.Series(results, name='Label')
submission = pd.concat([pd.Series(range(1,28001), name='ImageId'),results],axis = 1)
submission.to_csv("data/results/keras{}_submission.csv".format(noisiness),index=False)

print(">>> Submission saved under 'data/results/'")
