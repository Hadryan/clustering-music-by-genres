"""
Back Propogation Neural Network for optdigits
Sagar Gaur
SMAI Monsoon-2016
"""
import numpy as np
from numpy.linalg import norm
import random
import collections

from sklearn.preprocessing import minmax_scale

digit = {
    0: [1, 0, 0, 0],
    1: [0, 1, 0, 0],
    2: [0, 0, 1, 0],
    3: [0, 0, 0, 1],
}


def sigm(x):
    """ Sigmoid Function"""
    return 1. / (1. + np.exp(-x))


def delSigm(x):
    """ Derivative of Sigmoid Function"""
    return sigm(x) * (1. - sigm(x))


class NeuralNet:
    """
    Neural Network with backpropogation with
    1 input layer of size ni + 1,
    1 hidden layer of size nh
    1 output layer of size no
    """

    def __init__(self, ni, nh, no):
        # Initialise input, hidden and output units
        self.ni = ni + 1
        self.nh = nh
        self.no = no

        # create weights
        self.whi = [[0.] * self.nh for _ in xrange(self.ni)]
        self.woh = [[0.] * self.no for _ in xrange(self.nh)]
        # last change in weights for momentum
        self.chi = [[0.] * self.nh for _ in xrange(self.ni)]
        self.coh = [[0.] * self.no for _ in xrange(self.nh)]

        # Activation
        self.ai = [1.0] * self.ni
        self.ah = [1.0] * self.nh
        self.ao = [1.0] * self.no

        # Initialise with random weights
        for i in range(self.ni):
            for j in range(self.nh):
                self.whi[i][j] = 1.0 * random.random() - 0.5
        for j in range(self.nh):
            for k in range(self.no):
                self.woh[j][k] = 1.0 * random.random() - 0.5

    def update(self, input):
        """ Update values at nodes """

        # Input Layer
        for i in xrange(self.ni - 1):
            self.ai[i] = input[i]

        # First Hidden Layer
        for j in xrange(self.nh):
            sum = 0.
            for i in xrange(self.ni):
                sum += self.ai[i] * self.whi[i][j]
            self.ah[j] = sigm(sum)

        # Output Layer
        for k in xrange(self.no):
            sum = 0.
            for j in xrange(self.nh):
                sum += self.ah[j] * self.woh[j][k]
            self.ao[k] = sigm(sum)
        return self.ao

    def backPropogate(self, truth, _eta1, _eta2):
        """ Backpropogation with momentum """

        # Derivatives
        delOut = [0.] * self.no
        delHidden = [0.] * self.nh

        # Calculate errors for output units
        for k in xrange(self.no):
            error = truth[k] - self.ao[k]
            delOut[k] = delSigm(self.ao[k]) * error

        # Calculate errors for hidden units
        for j in xrange(self.nh):
            error = 0.
            for k in xrange(self.no):
                error += delOut[k] * self.woh[j][k]
            delHidden[j] = delSigm(self.ah[j]) * error

        # Update output weights
        for j in xrange(self.nh):
            for k in xrange(self.no):
                change = delOut[k] * self.ah[j]
                self.woh[j][k] += _eta1 * change + _eta2 * self.coh[j][k]
                self.coh[j][k] = change

        # Update input weights
        for i in xrange(self.ni):
            for j in xrange(self.nh):
                change = delHidden[j] * self.ai[i]
                self.whi[i][j] += _eta1 * change + _eta2 * self.chi[i][j]
                self.chi[i][j] = change

        error = sum([0.5 * (truth[k] - self.ao[k])**2 for i in
                     xrange(len(truth))])
        return error

    def printWeights(self):
        print "Input Weights"
        for i in xrange(self.ni):
            print self.whi[i]
        print "Output Weights"
        for j in xrange(self.nh):
            print self.woh[j]

    def train(self, data, testset, iterations=1000, eta1=0.1, eta2=0.1):
        """
        eta1 is the learning rate
        eta2 is momentum factor
        """
        for i in xrange(iterations):
            error = 0.
            for sample in data:
                input = sample[0]
                truth = sample[1]
                self.update(input)
                error += self.backPropogate(truth, eta1, eta2)
            if (i % 20) == 0:
                print "Iterations", i, '/', iterations
                print "error:", error
                self.test(testset)

    def test(self, samples):
        # print "Testing:\n"
        correct = 0
        total = 0
        conf_mat = [[0, 0, 0, 0] for i in range(len(digit))]
        for sample in samples:
            output = self.update(sample[0])
            # print output, sample
            label = 0
            dist = [abs(norm(np.array(output) - np.array(digit[i])))
                    for i in digit.keys()]
            label = dist.index(min(dist))
            total += 1
            if label == sample[2]:
                correct += 1
            conf_mat[int(label)][int(sample[2])] += 1
            # print "Expected:", int(sample[2]), "Output:", label
        print "Total:", total, "Correct:", correct
        print "Accuracy:", round(correct * 100. / total, 2), '%\n'
        for i in conf_mat:
            print i
        print ""


# def preprocess(data):
#     print data
#     reduced_data = []
#     for i in xrange(8):
#         sum = 0
#         for j in xrange(8):
#             for k in xrange(8):
#                 sum += data[j * 64 + k]
#         reduced_data.append[sum]
#     print reduced_data, len(reduced_data)


def main():
    filename = "data_0123.csv"
    file = np.loadtxt(filename, delimiter=',')
    X1 = []
    for i in file:
        j = list(i)
        X1.append(j)
    X1 = np.array(X1)
    X2 = minmax_scale(X1[:, range(len(X1[0]) - 1)],
                      feature_range=(0, 10), axis=0)
    X3 = np.concatenate((X2, X1[:, [-1]]), axis=1)
    data = []
    for j in X3:
        x = j[-1]
        j = j[: -1]
        y = digit[x]
        data.append([j, y, x])
    random.shuffle(data)
    # Divide into Training and Test data
    trainset = data[: int(0.7 * len(data))]
    testset = data[int(0.7 * len(data)):]
    print collections.Counter(np.array(testset).T[-1])
    for niter in [1200]:
        for i in [13]:
            print "Hidden Layers:", i, "Iterations:", niter
            nn = NeuralNet(11, i, 4)
            nn.train(trainset, testset, niter)
            # nn.test(testset)


if __name__ == "__main__":
    main()
