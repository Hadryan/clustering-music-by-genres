from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
from sklearn.linear_model import LogisticRegression
import random
from metric_learn import LMNN
from sklearn.metrics import accuracy_score

iris = load_iris()

f = open("data.csv", "r").read().split('\n')
random.shuffle(f)
X = []
y = []
for i in f:
    i = [float(j) for j in i.split(',')]
    X.append(i[:-1])
    y.append(i[-1])


X = np.array(X)[:500]
x = X
y = np.array(y)[:500]

print "Running on Reduced dataset of size 500"
def nearest_neighbors(X, y, neighbors=7):
    train = X[0: int(0.7 * len(X))]
    train_out = y[0: int(0.7 * len(y))]
    test = X[int(0.7 * len(X)):]
    test_out = y[int(0.7 * len(y)):]
    print "NN"
    for neighbors in [5, 7, 9]:
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
    LMNN(),
    # ITML_Supervised(num_constraints=200),
    # SDML_Supervised(num_constraints=200),
    # LSML_Supervised(num_constraints=200),
]

# x = train
# y = train_out

print "LMNN"
for ax_num, ml in enumerate(mls, start=3):
    print "Fitting"
    ml.fit(x, y)
    tx = ml.transform()
    # for i in tx:
    #     print i
    ml_knn = nearest_neighbors(tx, y)


# print "Normal"
nearest_neighbors(x,y)

train = X[0: int(0.7 * len(X))]
train_out = y[0: int(0.7 * len(y))]
test = X[int(0.7 * len(X)):]
test_out = y[int(0.7 * len(y)):]

# neighbors = 12

# knn = KNeighborsClassifier(n_neighbors=neighbors)

# knn.fit(train, train_out)

# out = knn.predict(test)
# correct = 0
# for key, val in enumerate(out):
#     if out[key] == test_out[key]:
#         correct += 1

# print "Accuracy", float(correct) / float(len(out)) * 100


# logreg = LogisticRegression()
# logreg.fit(train, train_out)

# out = logreg.predict(test)
# correct = 0
# for key, val in enumerate(out):
#     if out[key] == test_out[key]:
#         correct += 1

# print accuracy_score(test_out, out)

print "ANN"

from sklearn.neural_network import MLPClassifier

for layer in [13]:
    for iter in [1000]:
        print "max_iter", iter, "layer", layer
        mlp = MLPClassifier(hidden_layer_sizes=(
            layer), max_iter=iter, solver='lbfgs', verbose=False, early_stopping=False, learning_rate='constant')

        mlp.fit(train, train_out)
        out1 = mlp.predict(train)
        out2 = mlp.predict(test)
        print("Training set score: %f" % mlp.score(train, train_out))
        print("Test set score: %f" % mlp.score(test, test_out))

        x = [[0, 0, 0, 0] for k in range(4)]
        for key, val in enumerate(out1):
            x[int(val)][int(train_out[key])] += 1
        print x

        x = [[0, 0, 0, 0] for k in range(4)]
        for key, val in enumerate(out2):
            x[int(val)][int(test_out[key])] += 1
        print x
