from sklearn import svm
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy

iris = datasets.load_iris()
X = iris.data
y = iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=4)
#y_train = y_train.reshape(-1, 1)
y_test = y_test.reshape(-1, 1)
model = svm.SVC(kernel='linear')
model.fit(X_train, y_train)
y_predict = model.predict(X_test)
print(accuracy_score(y_test.ravel(), y_predict))