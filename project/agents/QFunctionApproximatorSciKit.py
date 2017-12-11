import numpy as np
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score

class QFunctionApproximatorSciKit(object):
    def __init__(self, player, numFeatures, batchSize=100, gamma=1, decay=0.99, alpha=0.1):
        self.player = player
        self.s, self.a, self.r = None, None, None
        self.weights = []
        self.batch = []
        self.batches = 0
        self.batchSize = batchSize
        self.gamma = gamma
        self.regr = linear_model.LinearRegression()
        X = np.array([np.ones(numFeatures) for i in range(batchSize)])
        Y = np.zeros(batchSize)
        print(X)
        print(Y)
        self.regr.fit(X, Y)

        self.alpha = alpha

    def Q(self, state, action):
        features = np.array(state.calculateFeatures(state, action, self.player))
        #print(self.regr.predict(np.array([features])))
        return self.regr.predict(np.array([features]))[0]

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

            self.batch.append([np.array(state.calculateFeatures(self.s, self.a, self.player)), q])

        if len(self.batch) == self.batchSize:
            X = np.array([b[0] for b in self.batch])
            Y = np.array([b[1] for b in self.batch])

            self.regr.fit(X, Y)
            self.weights = self.regr.coef_
            self.batches = []

    def finalize(self, state, reward, actions):
        self.updateBatch(state, reward, None)

def argmax(l):
    return l.index(max(l))
