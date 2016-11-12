from sklearn.neighbors import KNeighborsClassifier
import numpy as np
from sklearn.linear_model import LogisticRegression
import random
from metric_learn import LMNN
from sklearn.metrics import accuracy_score

f = open("data.csv", "r").read().split('\n')
random.shuffle(f)
X = []
y = []
for i in f:
    i = [float(j) for j in i.split(',')]
    X.append(i[:-1])
    y.append(i[-1])


X = np.array(X)
x = X
y = np.array(y)


def nearest_neighbors(X, y, neighbors=7):
    train = X[0: int(0.7 * len(X))]
    train_out = y[0: int(0.7 * len(y))]
    test = X[int(0.7 * len(X)):]
    test_out = y[int(0.7 * len(y)):]
    print "NN"
    for neighbors in [1, 3, 5, 7, 9, 11, 13, 15]:
        knn = KNeighborsClassifier(n_neighbors=neighbors)
        knn.fit(train, train_out)
        out = knn.predict(test)
        print "n=", neighbors, accuracy_score(test_out, out)
        x = [[0, 0, 0, 0] for i in range(4)]
        out1 = knn.predict(train)
        out2 = knn.predict(test)
        print("Training set score: %f" % knn.score(train, train_out))
        x = [[0, 0, 0, 0] for i in range(4)]
        for key, val in enumerate(out1):
            x[int(val)][int(train_out[key])] += 1
        print x
        print("Test set score: %f" % knn.score(test, test_out))
        x = [[0, 0, 0, 0] for i in range(4)]
        for key, val in enumerate(out2):
            x[int(val)][int(test_out[key])] += 1
        print x
    print "Logreg"
    logreg = LogisticRegression()
    logreg.fit(train, train_out)
    out1 = logreg.predict(train)
    out2 = logreg.predict(test)
    print("Training set score: %f" % logreg.score(train, train_out))
    x = [[0, 0, 0, 0] for i in range(4)]
    for key, val in enumerate(out1):
        x[int(val)][int(train_out[key])] += 1
    print x
    print("Test set score: %f" % logreg.score(test, test_out))
    x = [[0, 0, 0, 0] for i in range(4)]
    for key, val in enumerate(out2):
        x[int(val)][int(test_out[key])] += 1
    print x


mls = [
    LMNN()
]


print "LMNN"
for ax_num, ml in enumerate(mls, start=3):
    print "here"
    ml.fit(x, y)
    tx = ml.transform()
    ml_knn = nearest_neighbors(tx, y)
