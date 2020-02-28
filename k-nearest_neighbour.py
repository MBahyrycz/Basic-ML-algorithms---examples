import sklearn
from sklearn import preprocessing
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd


data = pd.read_csv("car.data")
# print(data.head())

le = preprocessing.LabelEncoder()
buying = le.fit_transform(list(data["buying"]))
maint = le.fit_transform(list(data["maint"]))
door = le.fit_transform(list(data["door"]))
persons = le.fit_transform(list(data["persons"]))
lug_boot = le.fit_transform(list(data["lug_boot"]))
safety = le.fit_transform(list(data["safety"]))
cls = le.fit_transform(list(data["class"]))

predict = "class"

X = list(zip(buying, maint, door, persons, lug_boot, safety))
Y = list(cls)

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, Y, test_size=0.1)

model = KNeighborsClassifier(n_neighbors=9)

model.fit(x_train, y_train)
acc = model.score(x_test, y_test)

print(acc)

predicted = model.predict(x_test)

names = ["unacc", "acc", "good", "vgood"]

for i in range(10):
# for i in range(len(predicted)):
    print("Predicted: ", names[predicted[i]], "Data: ", x_test[i], "Actually: ", names[y_test[i]])
    n = model.kneighbors([x_test[i]], 9)
    print("N", n)