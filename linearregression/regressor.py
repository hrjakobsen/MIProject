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
        number_of_weights = len(self.weights)
        new_weights = np.zeros(number_of_weights)

        for j in range(number_of_weights):
            new_weights[j] = self.weights[j] + self.alpha() \
                    * (sum([(self.targetFeature(d) - self.predict(d)) * self.features[j](d) / len(dataset) for d in dataset]))
        self.weights = new_weights

    def predict(self, values):
        values = np.asarray([feature(values) for feature in self.features])
        prediction = np.sum(np.dot(self.weights, values))
        return prediction

    def MSE(self, dataset):
        return sum([(self.targetFeature(data) - self.predict(data))**2 for data in dataset])/len(dataset)

    def alpha(self):
        self.called += 1
        alpha = 0.00001
        return alpha


def getDatapoints():
    return np.random.randint(0, 500, 50)


def main():
    r = Regressor([
        lambda x: 1,
        lambda x: x
    ],
        lambda x: 3*x+7
    )

    errs = []
    err = 1

    for _ in range(10000):
        p = getDatapoints()
        r.train(p)
        print(r.MSE(p))

    x = range(len(errs))
    y = errs

    print(r.MSE(getDatapoints()))
    print(r.weights)


if __name__ == "__main__":
    main()


