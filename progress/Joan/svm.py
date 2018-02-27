from sklearn import svm
from toolkit.file_operations import write_results, mnist_to_pdseries

[train, y, test] = mnist_to_pdseries("data/")

model = svm.SVC()
model.fit(train, y)

predictions = model.predict(test)
write_results(predictions, 'results/', 'svm_results')