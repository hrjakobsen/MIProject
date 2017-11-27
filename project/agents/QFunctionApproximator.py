import numpy as np


class QFunctionApproximator(object):
    def __init__(self, player, numFeatures, actions, batchSize=100, gamma=1):
        self.player = player
        self.weights = np.ones(numFeatures) * 0.1
        self.s, self.a, self.r = None, None, None
        self.actions = actions
        self.batch = []
        self.batches = 0
        self.batchSize = batchSize
        self.gamma = gamma
        self.mu = 0.999

        #Momentum
        self.velocity = np.zeros(numFeatures)

        #Cache for Adagrad and RMSprop
        self.g = np.zeros(numFeatures)

        #RMSprop
        self.decay = 0.9

    def Q(self, state, action):
        return np.sum(np.dot(state.calculateFeatures(state, action, self.player), self.weights))

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
            newWeights = self.weights
            differences = np.array([data[2] - self.Q(data[0], data[1]) for data in self.batch])
            calculatedFeatures = [state.calculateFeatures(data[0], data[1], self.player) for data in self.batch]
            gradients = np.dot(differences, calculatedFeatures)

            for j in range(len(self.weights)):
                gradient = gradients[j] / self.batchSize
                self.g[j] = self.decay * self.g[j] + (1 - self.decay ) * gradient ** 2
                #self.velocity[j] = self.mu * self.velocity[j] - gradient
                newWeights[j] -= self.alpha() * gradient / (np.sqrt(self.g[j]) + 0.0000001)
            self.weights = newWeights
            self.batch = []

        self.batches += 1

    def alpha(self):
        return 0.01

    def finalize(self, state, reward):
        self.updateBatch(state, reward)

    def minibatchGradientDescent(self, state):
        newWeights = []
        differences = [data[2] - self.Q(data[0], data[1]) for data in self.batch]
        for j in range(len(self.weights)):
            newWeights.append(self.weights[j] - self.alpha() * (1 / self.batchSize) * 2 * sum([differences[i] * (state.getFeatures(self.player)[j](data[0], data[1])) for i, data in enumerate(self.batch)]))
        self.weights = np.asarray(newWeights)
        self.batch = []
        self.batches += 1

    def momentumGradientDescent(self, state):
        newVelocity = []
        differences = [data[2] - self.Q(data[0], data[1]) for data in self.batch]
        for j in range(len(self.velocity)):
            newVelocity.append(self.learningVelocity * self.velocity[j] + self.alpha() * (1 / self.batchSize) * 2 * sum([differences[i] * (state.getFeatures(self.player)[j](data[0], data[1])) for i, data in enumerate(self.batch)]))

        for j in range(len(newVelocity)):
            self.weights[j] -= newVelocity[j]

        self.velocity = newVelocity
        self.batch = []
        self.batches += 1


def argmax(l):
    return l.index(max(l))
