"""
Linear Discriminant Functions
Sagar Gaur
SMAI Monsoon-2016
"""
import numpy as np
import matplotlib.pyplot as plt
import pylab as pl


w1 = [(2, 7), (8, 1), (7, 5), (6, 3), (7, 8), (5, 9), (4, 5)]
w2 = [(4, 2), (-1, -1), (1, 3), (3, -2), (5, 3.25), (2, 4), (7, 1)]
_eta = 1.0
MAX_ITER = 99999


def singleSamplePptron(sample, b=0., w=0.):
    """
    Single Sample Percepton
    LL: a(i+1) = a(i) + y(i)
    Stop when all samples are correctly classified
    """
    a = np.array([1., 1., 1.]) * w
    # number of correctly classified samples
    i = 0  # index
    c = 0  # Correctly classified
    iter = MAX_ITER  # Halt at Max Iterations
    while iter > 0 and c <= len(sample):
        if np.dot(sample[i], a.T) <= b:
            a += sample[i]
            c = 0
        else:
            c += 1
        i = (i + 1) % len(sample)
        iter -= 1
    print MAX_ITER - iter
    return a, MAX_ITER - iter


def marginRelax(sample, b=0., w=0.):
    """
    Batch Relaxation with Margin
    LL: a(i+1) = a(i) + _eta * del(J)
    del(J) = _sigma (b-a.T)*y/mod(y)^2
    """
    a = np.array([1., 1., 1.]) * w
    i = 0
    c = 0
    iter = MAX_ITER
    while iter > 0 and c <= len(sample):
        j = sample[i]
        if np.dot(j, a.T) < b:
            delJ = (b - np.dot(a.T, j)) * j / np.dot(j, j)
            a += _eta * delJ
            if np.linalg.norm(delJ) <= 0.001:
                c += 1
            else:
                c = 0
        i = (i + 1) % len(sample)
        iter -= 1
    print MAX_ITER - iter
    return a, MAX_ITER - iter


def lms(sample, b):
    """
    Least Mean Squared Procedure
    LL: a(i+1) = a(i) + _eta * y(i) * e(i)
    e(i) = b(i) - a.T * y(i)
    """
    i = 0  # index
    # c = 0
    # eta = _eta
    # threshold = 0.00001
    b = np.random.rand(1, len(sample))[0] * 20 - 10
    a = np.array([1., 1., 1.])
    iter = MAX_ITER
    while iter > 0:
        j = sample[i]
        # print np.dot(a.T, j), j
        # print a, b, sample
        if np.dot(a.T, j) < j[0]:
            delJ = (j[0] - np.dot(a.T, j)) * j / np.dot(j, j)
            # print j
            a -= _eta * delJ
            # if np.linalg.norm(delJ) <= 0.001:
                # break
        iter -= 1
        # hypothesis = np.dot(sample, a.T) - b
        # loss = b - hypothesis.T
        # gradient = np.dot(loss, sample)
        # a -= eta * gradient / np.linalg.norm
        # # Error Value
        # j += 1
        # print gradient
        i = (i + 1) % len(sample)
        # eta /= j  # annealing
        # if np.linalg.norm(eta * gradient) < threshold:
        #     break
    return a


def makeSample(c1, c2):
    c1 = [np.array([1, 1, i[0], i[1]]) for i in c1]
    c2 = [np.array([-1, -1, i[0], i[1]]) for i in c2]
    sample = np.concatenate((c1, c2))
    return sample.T[1:].T, sample.T[0]


def plot(A, title=""):
    color = ['g', 'c', 'm', 'y']
    label = ['SSP', 'SSPM', 'MR']
    x1 = [i[0] for i in w1]
    y1 = [i[1] for i in w1]
    x2 = [i[0] for i in w2]
    y2 = [i[1] for i in w2]
    plt.plot(x1, y1, 'ro')
    plt.plot(x2, y2, 'bo')
    for i, a in enumerate(A):
        x = [-3, 10]
        y = [-1. * (a[0] + a[1] * j) / a[2] for j in x]
        plt.plot(x, y, color[i] + '--', label=label[i])
        # Plot Vector
        pl.arrow(0, 0, a[1], a[2], fc=color[i], ec=color[i],
                 head_width=0.15, head_length=0.2)
    plt.legend(loc=4, borderaxespad=0.)
    plt.title(title)
    # plt.axis([-5, 15, -5, 15])
    plt.show()
    # plt.savefig(title + ".png")
    # plt.clf()


def main():
    sample, label = makeSample(w1, w2)
    iter = []
    a_lms = lms(sample, label)
    A = [a_lms]
    plot(A, "")


    # for m in [0., 1., 4., 8., 16., 32., 64.]:
    #     iterw = []
    #     for w in [0., 1., 4., 8., 16., 32., 64.]:
    #         # print "weight, margin: ", w, m
    #         # a_ssp, iter1 = singleSamplePptron(sample, 0., w)
    #         # a_sspm, iter2 = singleSamplePptron(sample, m, w)
    #         # a_mr, iter3 = marginRelax(sample, m, w)
    #         # iterw.append([iter1, iter2, iter3])
    #         a_lms = lms(sample, label)
    #         # print a_ssp
    #         # print a_sspm
    #         # print a_mr
    #         # A = [a_ssp, a_sspm, a_mr]
    #         # A = [a_lms]
    #         # print a_lms
    #         # plot(a_ssp, "Single Sample Perceptron with Weight = " + str(w))
    #         # plot(a_sspm, "Single Sample Perceptron with Weight = " +
    #         #      str(m) + " Boundary = " + str(w))
    #         # plot(a_mr, "Relaxation with Margin = " +
    #         #      str(m) + " Weight = " + str(w))
    #         # plot(a_lms)
    #         # plot(A, "Margin = " + str(m) + " Weight = " + str(w))
    #     iter.append(iterw)
    # print iter


if __name__ == "__main__":
    main()
