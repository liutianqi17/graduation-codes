from sklearn import svm
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.externals import joblib

def show_accuracy(y_hat, y_test, param):
    pass

path = r'C:\Users\Neticle\Desktop\bishedaima\Fusing Multiple\prob_data.txt'
data = np.loadtxt(path, dtype=float, delimiter=',', encoding='utf-8-sig')

x, y = np.split(data, (4,), axis=1)
x = x[:, :5]
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=1, train_size=0.6)

clf = svm.SVC(C=0.8, kernel='rbf', gamma=20, decision_function_shape='ovr')
clf.fit(x_train, y_train.ravel())
joblib.dump(clf, "svm_model.m")

print(clf.score(x_train, y_train))
y_hat = clf.predict(x_train)
show_accuracy(y_hat, y_train, '训练集')
print(clf.score(x_test, y_test))
y_hat = clf.predict(x_test)
show_accuracy(y_hat, y_test, '测试集')
