from sklearn.ensemble import RandomForestClassifier
from toolkit.file_operations import write_results, mnist_to_pdseries

[train, y, test] = mnist_to_pdseries("data/")

model = RandomForestClassifier()
model.fit(train, y)

predictions = model.predict(test)
write_results(predictions, 'results/', 'random_forest_results')