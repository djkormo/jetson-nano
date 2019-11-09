from sklearn import datasets
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier
# load the iris datasets
print("Loading Iris dataset")
dataset = datasets.load_iris()
# fit a CART model to the data
print("Fitting decision tree model")
model = DecisionTreeClassifier()
model.fit(dataset.data, dataset.target)
print(model)
# make predictions
print ("Making predictions")
expected = dataset.target
predicted = model.predict(dataset.data)
# summarize the fit of the model
print(metrics.classification_report(expected, predicted))
print ("Showing confusion matrix")
print(metrics.confusion_matrix(expected, predicted))
