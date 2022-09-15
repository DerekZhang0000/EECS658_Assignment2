from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

from sklearn.base import clone
from sklearn.metrics import confusion_matrix
import numpy as np
import pandas as pd

# Reads data from csv
df = pd.read_csv("iris.csv", header=None)
# Shuffles data
df = df.iloc[np.random.RandomState(seed=0).permutation(len(df))]

# X is a list of lists. The inner list is comprised of the characteristics of the corresponding flower type.
X = df.drop(df.columns[[4]], axis=1).to_numpy()
# Y is a list of flower types that correspond to a list of characteristics in X
Y = df.transpose().iloc[4].to_numpy()
YInt = Y
YInt[YInt == "Iris-setosa"] = 0
YInt[YInt == "Iris-versicolor"] = 1
YInt[YInt == "Iris-virginica"] = 2

# Splits sample in half
Fold1_X = X[:75]
Fold1_Y = Y[:75]
Fold1_YInt = YInt[:75]
Fold2_X = X[75:]
Fold2_Y = Y[75:]
Fold2_YInt = YInt[75:]

# Initializing classifiers
LRClf = LinearRegression()
poly2Clf = LinearRegression()
poly3Clf = LinearRegression()
NBClf = GaussianNB()
KNNClf = KNeighborsClassifier()
LDAClf = LinearDiscriminantAnalysis()
QDAClf = QuadraticDiscriminantAnalysis()

# Linear Regression Classifier
print("Linear Regression Classifier")
LRClf2 = clone(LRClf)
LRClf.fit(Fold1_X, Fold1_Y)
LRClf2.fit(Fold2_X, Fold2_Y)
predictions = np.rint(np.concatenate((LRClf.predict(Fold1_X), LRClf2.predict(Fold2_X)))).astype('int')
concatArr = np.concatenate((Fold1_YInt, Fold2_YInt)).astype('int')
matrix = confusion_matrix(concatArr, predictions, labels=[0, 1, 2])
score = (LRClf.score(Fold1_X, Fold1_Y) + LRClf2.score(Fold2_X, Fold2_Y)) / 2
print("Score:", score)
print("Matrix:\n", matrix, "\n")

# Polynomial Regression Classifiers
print("Polynomial 2nd-Degree Regression Classifier")
poly2 = PolynomialFeatures(degree=2, include_bias=False)
poly2Clf2 = clone(poly2Clf)
poly2Clf.fit(poly2.fit_transform(Fold1_X), Fold1_YInt)
poly2Clf2.fit(poly2.fit_transform(Fold2_X), Fold2_YInt)
predictions = np.rint(np.concatenate((poly2Clf.predict(poly2.transform(Fold1_X)), poly2Clf2.predict(poly2.transform(Fold2_X))))).astype('int')
concatArr = np.concatenate((Fold1_YInt, Fold2_YInt)).astype('int')
matrix = confusion_matrix(concatArr, predictions, labels=[0, 1, 2])
score = (poly2Clf.score(poly2.transform(Fold1_X), Fold1_Y) + poly2Clf2.score(poly2.transform(Fold2_X), Fold2_Y)) / 2
print("Score:", score)
print("Matrix:\n", matrix, "\n")

print("Polynomial 3rd-Degree Regression Classifier")
poly3 = PolynomialFeatures(degree=3, include_bias=False)
poly3Clf2 = clone(poly3Clf)
poly3Clf.fit(poly3.fit_transform(Fold1_X), Fold1_YInt)
poly3Clf2.fit(poly3.fit_transform(Fold2_X), Fold2_YInt)
predictions = np.rint(np.concatenate((poly3Clf.predict(poly3.transform(Fold1_X)), poly3Clf2.predict(poly3.transform(Fold2_X))))).astype('int')
concatArr = np.concatenate((Fold1_YInt, Fold2_YInt)).astype('int')
matrix = confusion_matrix(concatArr, predictions, labels=[0, 1, 2])
score = (poly3Clf.score(poly3.transform(Fold1_X), Fold1_Y) + poly3Clf2.score(poly3.transform(Fold2_X), Fold2_Y)) / 2
print("Score:", score)
print("Matrix:\n", matrix, "\n")

# Naive Bayesian Classifier
print("Naive Bayesian Analysis Classifier")
NBClf2 = clone(NBClf)
NBClf.fit(Fold1_X, Fold1_YInt.astype('int'))
NBClf2.fit(Fold2_X, Fold2_YInt.astype('int'))
predictions = np.rint(np.concatenate((NBClf.predict(Fold1_X), NBClf2.predict(Fold2_X)))).astype(int)
concatArr = np.concatenate((Fold1_YInt, Fold2_YInt)).astype(int)
matrix = confusion_matrix(concatArr, predictions, labels=[0, 1, 2])
score = (NBClf.score(Fold1_X, Fold1_YInt.astype('float')) + NBClf2.score(Fold2_X, Fold2_YInt.astype('float'))) / 2
print("Score:", score)
print("Matrix:\n", matrix, "\n")

# K-Nearest Neighbors Classifier
print("K-Nearest Neighbors Classifier")
KNNClf2 = clone(KNNClf)
KNNClf.fit(Fold1_X, Fold1_YInt.astype('int'))
KNNClf2.fit(Fold2_X, Fold2_YInt.astype('int'))
predictions = np.rint(np.concatenate((KNNClf.predict(Fold1_X), KNNClf2.predict(Fold2_X)))).astype(int)
concatArr = np.concatenate((Fold1_YInt, Fold2_YInt)).astype(int)
matrix = confusion_matrix(concatArr, predictions, labels=[0, 1, 2])
score = (KNNClf.score(Fold1_X, Fold1_YInt.astype('float')) + KNNClf2.score(Fold2_X, Fold2_YInt.astype('float'))) / 2
print("Score:", score)
print("Matrix:\n", matrix, "\n")

# Linear Discriminant Analysis Classifier
print("Linear Discriminant Analysis Classifier")
LDAClf2 = clone(LDAClf)
LDAClf.fit(Fold1_X, Fold1_YInt.astype('int'))
LDAClf2.fit(Fold2_X, Fold2_YInt.astype('int'))
predictions = np.rint(np.concatenate((LDAClf.predict(Fold1_X), LDAClf2.predict(Fold2_X)))).astype(int)
concatArr = np.concatenate((Fold1_YInt, Fold2_YInt)).astype(int)
matrix = confusion_matrix(concatArr, predictions, labels=[0, 1, 2])
score = (LDAClf.score(Fold1_X, Fold1_YInt.astype('float')) + LDAClf2.score(Fold2_X, Fold2_YInt.astype('float'))) / 2
print("Score:", score)
print("Matrix:\n", matrix, "\n")

# Quadtratic Discriminant Analysis Classifier
print("Quadratic Discriminant Analysis Classifier")
QDAClf2 = clone(QDAClf)
QDAClf.fit(Fold1_X, Fold1_YInt.astype('int'))
QDAClf2.fit(Fold2_X, Fold2_YInt.astype('int'))
predictions = np.rint(np.concatenate((QDAClf.predict(Fold1_X), QDAClf2.predict(Fold2_X)))).astype(int)
concatArr = np.concatenate((Fold1_YInt, Fold2_YInt)).astype(int)
matrix = confusion_matrix(concatArr, predictions, labels=[0, 1, 2])
score = (QDAClf.score(Fold1_X, Fold1_YInt.astype('float')) + QDAClf2.score(Fold2_X, Fold2_YInt.astype('float'))) / 2
print("Score:", score)
print("Matrix:\n", matrix, "\n")