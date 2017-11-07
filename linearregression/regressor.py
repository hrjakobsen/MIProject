import numpy as np
import matplotlib.pyplot
import math
import pylab
np.random.seed(0)


class Regressor(object):
    def __init__(self, features, target_feature):
        self.targetFeature = target_feature
        self.features = features
        self.weights = np.asarray([0.5, 0.5])
        self.called = 0

    def train(self, dataset):
        for example in dataset:
            error = self.targetFeature(example) - self.predict(example)
            update = self.alpha()*error
            for j in range(len(self.weights)):
                self.weights[j] = self.weights[j] + update * self.features[j](example)

    def predict(self, values):
        values = np.asarray([feature(values) for feature in self.features])
        prediction = np.sum(np.dot(self.weights, values))
        return prediction

    def MSE(self, dataset):
        return sum([(self.targetFeature(data) - self.predict(data))**2 for data in dataset])/len(dataset)

    def alpha(self):
        self.called += 1
        alpha = 0.0001
        return alpha


def getDatapoints():
    return np.random.randint(0, 5, 1000)


def main():
    r = Regressor([
        lambda x: 1,
        lambda x: x
    ],
        lambda x: 3*x+7
    )

    errs = []
    err = 1



    r.train(getDatapoints())
    r.train(getDatapoints())
    r.train(getDatapoints())
    r.train(getDatapoints())
    r.train(getDatapoints())
    r.train(getDatapoints())


    x = range(len(errs))
    y = errs

    # matplotlib.pyplot.plot(x, y, '-o')
    # matplotlib.pyplot.show()

    print(r.MSE(getDatapoints()))
    print(r.weights)


if __name__ == "__main__":
    main()


