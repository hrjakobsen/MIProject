import numpy as np


class QFunctionApproximator(object):
    def __init__(self, player, numFeatures, actions, batchSize=1, gamma=1):
        self.player = player
        self.weights = np.ones(numFeatures) * 0.5
        self.s, self.a, self.r = None, None, None
        self.actions = actions
        self.batch = []
        self.batches = 0
        self.batchSize = batchSize
        self.gamma = gamma

    def Q(self, state, action):
        features = np.asarray([feature(state, action) for feature in state.getFeatures(self.player)])
        return np.sum(features * self.weights)

    def getMove(self, state, reward):
        self.updateBatch(state, reward)

        a = self.actions[argmax([self.Q(state, aP) for aP in self.actions])]

        self.s = state
        self.a = a

        return self.a

    def updateBatch(self, state, reward):
        if self.s is not None:
            # This is the q that Q would become if we were not using the function approximation method but instead used the tabular method. -Tessa
            q = (1 - self.alpha()) * self.Q(self.s, self.a) + self.alpha() * (reward + self.gamma * max([self.Q(state, aP) for aP in self.actions]))
            self.batch.append((self.s, self.a, q))

        if len(self.batch) == self.batchSize:
            # Update weights
            newWeights = []
            differences = [data[2] - self.Q(data[0], data[1]) for data in self.batch]
            for j in range(len(self.weights)):
                newWeights.append(self.weights[j] - self.alpha() * (1 / self.batchSize) * 2 * sum([differences[i] * (-state.getFeatures(self.player)[j](data[0], data[1])) for i, data in enumerate(self.batch)]))
            self.weights = np.asarray(newWeights)
            self.batch = []

        self.batches += 1

    def alpha(self):
        return 60 / (60 + self.batches)

    def finalize(self, state, reward):
        self.updateBatch(state, reward)


def argmax(l):
    return l.index(max(l))