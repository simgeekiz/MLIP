
import warnings; warnings.simplefilter('ignore')
import pickle
import pandas as pd
# from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.feature_selection import SelectPercentile, chi2
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
training_data = pd.read_csv('train.csv')
training_data = training_data[:200]
# len(training_data)
y = training_data['label']
X = training_data.loc[:, training_data.columns != 'label']
# Feature Selection
feature_selector = SelectPercentile(chi2, percentile=70)
feature_selector.fit(X, y)

list_of_new_features = []
for i, feature_name in enumerate(list(X)):
    if feature_selector.get_support()[i]:
        list_of_new_features.append(feature_name)

X = feature_selector.transform(X)
X = pd.DataFrame(X, columns=list_of_new_features)

# Classifier Comparison

names = ["Nearest Neighbors", "Linear SVM", "RBF SVM",# "Gaussian Process",
         "Decision Tree", "Random Forest", "Neural Net", "AdaBoost",
         "Naive Bayes", "QDA"]

classifiers = [
    KNeighborsClassifier(3),
    SVC(kernel="linear", C=0.025),
    SVC(gamma=2, C=1),
#     GaussianProcessClassifier(1.0 * RBF(1.0)),
    DecisionTreeClassifier(max_depth=5),
    RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
    MLPClassifier(alpha=1),
    AdaBoostClassifier(),
    GaussianNB(),
    QuadraticDiscriminantAnalysis()]

prediction_per_classifier_df = pd.DataFrame()
for name, classifier in zip(names, classifiers):
    y_true, y_pred = y, cross_val_predict(classifier, X, y)
    prediction_per_classifier_df[name] = y_pred
    prediction_per_classifier_df['truelabel'] = y_true

prediction_per_classifier_df.to_csv('prediction_per_classifier_df.csv')

new_y_pred = []
for index, row in prediction_per_classifier_df.iterrows():
    most_likely_dict= {}
    for name in names:
        most_likely_dict[name] = row[name]
    most_likely_pred = max(set(list(most_likely_dict.values())),   key = list(most_likely_dict.values()).count)
    new_y_pred.append(most_likely_pred)



y_true = prediction_per_classifier_df['truelabel']
print('\n--- Ensemble ---')
print('Precision:', precision_score(y_true, new_y_pred, average='weighted'))
print('Recall:', recall_score(y_true, new_y_pred, average='weighted'))
print('F1-score:', f1_score(y_true, new_y_pred, average='weighted'))
print('Accuracy:', accuracy_score(y_true, new_y_pred))

# for name, classifier in zip(names, classifiers):
#
#     y_true, y_pred = y, cross_val_predict(classifier, X, y)
#
#     print('\n---', name, '---')
#     print('Precision:', precision_score(y_true, y_pred, average='weighted'))
#     print('Recall:', recall_score(y_true, y_pred, average='weighted'))
#     print('F1-score:', f1_score(y_true, y_pred, average='weighted'))
#     print('Accuracy:', accuracy_score(y_true, y_pred))
