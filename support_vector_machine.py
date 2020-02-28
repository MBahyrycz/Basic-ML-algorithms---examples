import sklearn
from sklearn import datasets
from sklearn import svm
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier

cancer = datasets.load_breast_cancer()

# print(cancer.feature_names)
# print(cancer.target_names)

x = cancer.data
y = cancer.target

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.2)

classes = ['malignant', 'benign']

### svm implementation
clf = svm.SVC(kernel="linear", C=2)
clf.fit(x_train, y_train)
y_predict1 = clf.predict(x_test)
acc1 = metrics.accuracy_score(y_test, y_predict1)

### knn implementation
knn = KNeighborsClassifier(n_neighbors=7)
knn.fit(x_train, y_train)
y_predict2 = knn.predict(x_test)
acc2 = metrics.accuracy_score(y_test, y_predict2)

### checking which accuracy is better
print("Acc for svm: {0} \nAcc for knn: {1}".format(acc1, acc2))

