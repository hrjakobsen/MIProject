import numpy as np
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score

class QFunctionApproximatorSciKit(object):
    def __init__(self, player, numFeatures, batchSize=100, gamma=1, decay=0.99, alpha=0.1):
        self.player = player
        #self.weights = np.ones(numFeatures) * np.random.randint(-10, 10)
        self.s, self.a, self.r = None, None, None
        self.batch = []
        self.batches = 0
        self.batchSize = batchSize
        self.gamma = gamma
        self.regr = linear_model.LinearRegression()
        self.regr.fit(np.ones(numFeatures), np.zeros(numFeatures))

        self.alpha = alpha

    def Q(self, state, action):
        return self.regr.predict(state.calculateFeatures(state, action, self.player))

    def getMove(self, state, reward, actions):
        self.updateBatch(state, reward, actions)

        a = actions[argmax([self.Q(state, aP) for aP in actions])]

        self.s = state
        self.a = a

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
            self.regr.fit([self.Q(data[0], data[1]) for data in self.batch], self.batch[2])
            #self.weights = self.regr.coef_
            print(self.regr.coef_)

        self.batches += 1

    def finalize(self, state, reward):
        self.updateBatch(state, reward, None)

def argmax(l):
    return l.index(max(l))
