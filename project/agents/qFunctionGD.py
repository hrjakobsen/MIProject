import numpy as np
import copy
import random
from interfaces import IAgent
from interface import implements


class QFunctionGD(implements(IAgent)):
    def __init__(self, player, numFeatures, batchSize=100, gamma=1, alpha=0.1, decay=0.99, minWeight=0, maxWeight=0):
        """
        :param player: the id of the player this agent plays as
        :param numFeatures: number of features the agent has access to
        :param batchSize: number of moves to perform between each update
        :param gamma: discount factor to use
        :param alpha: learning rate to use
        :param decay: decay rate used to shrink RMSprop's cache
        :param minWeight: minimum start weight
        :param maxWeight: maximum start weight
        """
        self.player = player
        self.numFeatures = numFeatures
        self.weights = np.random.uniform(minWeight, maxWeight, size=numFeatures)
        self.s, self.a, self.r = None, None, None
        self.batch = []
        self.batches = 0
        self.batchSize = batchSize
        self.gamma = gamma
        self.alpha = alpha
        self.q = None

        # Momentum
        self.velocity = np.zeros(numFeatures)
        self.mu = 0.999

        # Cache for ADAGRAD and RMSprop
        self.g = np.zeros(numFeatures)

        # RMSprop
        self.decay = decay

        random.seed(0)

    def Q(self, state, action):
        return np.sum(np.dot(state.getFeatures(self.player, action), self.weights))

    def getMove(self, state):
        reward = state.getReward(self.player)
        actions = state.getActions(self.player)

        self.updateBatch(state, reward, actions)

        self.s = copy.deepcopy(state)
        self.a = actions[argmax([self.Q(state, aP) for aP in actions])]

        return self.a

    def getTrainedMove(self, state):
        actions = state.getActions(self.player)
        return actions[argmax([self.Q(state, aP) for aP in actions])]

    def finalize(self, state):
        self.updateBatch(state, state.getReward(self.player), None)

    def getInfo(self):
        return self.weights

    def updateBatch(self, state, reward, actions):
        if self.s is not None:
            if actions is None:
                q = (1 - self.alpha) * self.Q(self.s, self.a) + self.alpha * reward
            else:
                q = (1 - self.alpha) * self.Q(self.s, self.a) + self.alpha * (reward + self.gamma * max([self.Q(state, aP) for aP in actions]))

            self.batch.append((self.s, self.a, q))

        if len(self.batch) == self.batchSize:
            # Update weights
            newWeights = self.weights
            differences = np.array([data[2] - self.Q(data[0], data[1]) for data in self.batch])
            calculatedFeatures = [-data[0].getFeatures(self.player, data[1]) for data in self.batch]
            gradients = np.dot(differences, calculatedFeatures)

            for j in range(self.numFeatures):
                gradient = gradients[j] / self.batchSize

                # Vanilla
                # newWeights[j] -= self.alpha * gradient

                # Momentum
                #self.velocity[j] = self.mu * self.velocity[j] + gradient
                #newWeights[j] -= self.alpha * self.velocity[j]

                # ADAGRAD
                # self.g[j] += gradient ** 2
                # newWeights[j] -= self.alpha * gradient / (np.sqrt(self.g[j]) + 0.0000001)

                # RMSProp
                self.g[j] = (self.decay * self.g[j]) + ((1 - self.decay) * gradient ** 2)
                newWeights[j] -= self.alpha * gradient / (np.sqrt(self.g[j]) + 0.000000001)

            self.weights = newWeights
            self.batch = []


def argmax(l):
    return l.index(max(l))