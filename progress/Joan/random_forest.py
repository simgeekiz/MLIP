from sklearn.ensemble import RandomForestClassifier
from toolkit.file_operations import write_results, mnist_to_pdseries

import toolkit.noise as noise

[train, y, test] = mnist_to_pdseries("data/")

model = RandomForestClassifier()
model.fit(train, y)

predictions = model.predict(test)

output_name = input('Enter name for outputfile\n')

write_results(predictions, 'results/', output_name)