import numpy as np

class QFunctionApproximator(object):
    def __init__(self, player, numFeatures, batchSize=100, gamma=1, decay=0.99, alpha=0.1, minWeight=0, maxWeight=0):
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

        # Cache for Adagrad and RMSprop
        self.g = np.zeros(numFeatures)

        # RMSprop
        self.decay = decay

    def Q(self, state, action):
        return np.sum(np.dot(state.calculateFeatures(state, action, self.player), self.weights))

    def getMove(self, state, reward, actions):
        self.updateBatch(state, reward, actions)

        self.s = state
        self.a = actions[argmax([self.Q(state, aP) for aP in actions])]

        return self.a

    def getTrainedMove(self, state, actions):
        return actions[argmax([self.Q(state, aP) for aP in actions])]

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
            calculatedFeatures = [-state.calculateFeatures(data[0], data[1], self.player) for data in self.batch]
            gradients = np.dot(differences, calculatedFeatures)

            for j in range(self.numFeatures):
                gradient = gradients[j] / self.batchSize

                # Vanilla
                # newWeights[j] -= self.alpha * gradient

                # Momentum
                # self.velocity[j] = self.mu * self.velocity[j] + gradient
                # newWeights[j] -= self.alpha * self.velocity[j]

                # ADAGRAD
                self.g[j] += gradient ** 2
                newWeights[j] -= self.alpha * gradient / (np.sqrt(self.g[j]) + 0.0000001)

                # RMSProp
                # self.g[j] = (self.decay * self.g[j]) + ((1 - self.decay) * gradient ** 2)
                # newWeights[j] -= self.alpha * gradient / (np.sqrt(self.g[j]) + 0.0000001)

            self.weights = newWeights
            self.batch = []

    def finalize(self, state, reward, actions):
        self.updateBatch(state, reward, None)

def argmax(l):
   return l.index(max(l))