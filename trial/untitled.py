#!/usr/bin/python
import numpy as np
import random
import matplotlib.pyplot as plt
import pylab as pl


def singleSampleP(sample):
    # initialize weight vector
    w = np.array([[0], [0], [0]])
    i = 0
    while i < len(sample):
        dp = np.dot(w.transpose(), sample[i])
        if dp[0][0] <= 0:
            # update weight vector
            w = w + sample[i]
            i = 0
        else:
            i += 1

    return w


def singleSamplePWithMargin(sample, b):
    # initialize weight vector
    w = np.array([[0], [0], [0]])
    i = 0
    while i < len(sample):
        dp = np.dot(w.transpose(), sample[i])
        if dp[0][0] <= b:
            # update weight vector
            w = w + sample[i]
            i = 0
        else:
            i += 1

    return w


def relax_with_margin(sample, b):
    # initialize weight vector
    w = np.array([[1], [1], [1]])
    i = 0

    while i < len(sample):
        # j += 1
        dp = np.dot(w.transpose(), sample[i])
        if dp[0][0] <= b:
            # update weight vector
            iv = sample[i]
            sum_of_squares = sum([j[0] * j[0] for j in iv])
            print "sum_of_squares: ", sum_of_squares
            k = float((b - dp[0][0])) / (sum_of_squares)
            print "k is : ", k
            print "dp is: ", dp  # , numerator
            deltaW = k * iv
            print "delta w is: ", deltaW
            w = w + 1.9 * deltaW
            i = 0
            print "w is : ", w
            # break
        else:
            i += 1
    return w


def lms(sample, label):
    # initialize weight vector
    w = np.array([[1], [1], [1]])
    i = 0
    j = 1
    # while j < 1000:
    while i < len(sample):
        dp = np.dot(w.transpose(), sample[i])
        key = (sample[i][1][0], sample[i][2][0])
        err = label[key] - dp[0][0]
        print "dp is : ", dp
        print "err is : ", err
        check = 1.8 * (err * sample[i])

        if np.linalg.norm(check) < 1:  # need to fix
            # update weight vector
            w = w + 0.8 * (err * sample[i])
            i = 0
        else:
            i += 1
        j += 1
        if j == 10000:
            break
    return w


def preprocessing(classA, classB):
    # return a list containg all items
    sample = [np.array([[1], [item[0]], [item[1]]]) for item in classA]
    for item in classB:
        sample.append(np.array([[1], [item[0]], [item[1]]]) * -1)
    # random.shuffle(sample)
    return sample


def assign_class(classA, classB):
    label = {}
    for item in classA:
        s = (item[0], item[1])
        label[s] = 1  # taking classA as 1
    for item in classB:
        s = (-item[0], -item[1])
        label[s] = 0  # taking classB as 0
    return label


def plot_graph(classA, classB, wvector, title, b=2):
    #fig, axis = plt.sublplot(nrows=2, ncols=2)
    yA = []
    xA = []
    yB = []
    xB = []
    for instA, instB in zip(classA, classB):
        xA.append(instA[0])
        yA.append(instA[1])
        xB.append(instB[0])
        yB.append(instB[1])
    #ax = axis[0, 0]
    plt.plot(xA, yA, 'ro')
    plt.plot(xB, yB, 'bo')
    #plt.plot([0, 3], [0, 2], linestyle="dashed", marker="o", color="green")
    # plot weight vector
    xW = (0, wvector[1])
    yW = (0, wvector[2])
    # print wvector[1], wvector[2]
    # plt.plot(xW, yW, label="weight", linestyle="-", marker="o", color="green")
    pl.arrow(0, 0, wvector[1][0], wvector[2][0], fc="g",
             ec="g", head_width=0.15, head_length=0.2)

    # plot hyperplane
    #c, a, b = w[0][1], w[1][1], w[2][1]
    # find y which is X2, assuming X1=0, 10
    y1 = -float((wvector[0][0])) / wvector[2][0]
    y2 = -float((wvector[1][0] * 10 + wvector[0][0])) / wvector[2][0]

    y1r = float((b - wvector[0][0])) / wvector[2][0]
    y2r = float(((b - wvector[0][0]) - wvector[1][0] * 10)) / wvector[2][0]

    # plt.plot((0, 10), (y1r, y2r), "c--") # Wtranspose.Y = b
    plt.plot((0, 10), (y1, y2), "m--")

    plt.axis([0, 15, 0, 15])
    plt.title(title)
    plt.draw()
    plt.show()


w1 = [(2, 7), (8, 1), (7, 5), (6, 3), (7, 8), (5, 9), (4, 5)]
w2 = [(4, 2), (-1, -1), (1, 3), (3, -2), (5, 3.25), (2, 4), (7, 1)]

sample = preprocessing(w1, w2)
print sample
label = assign_class(w1, w2)
print label

wvector = singleSampleP(sample)
# print wvector
w = singleSamplePWithMargin(sample, 15)
print wvector
print w
# wm = relax_with_margin(sample, 2)
# print wm
#wlms = lms(sample, label)

plot_graph(w1, w2, wvector, "single sample perceptron")
plot_graph(w1, w2, w, "single sample perceptron with margin", 2)
# plot_graph(w1, w2, wm, "relaxation procedure with margin", 2)
#plot_graph(w1, w2, wlms)

#plot_graph3D(w1, w2, wvector)
