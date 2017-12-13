import numpy as np
from interfaces import IAgent
from interface import implements
from keras.models import Sequential
from keras.layers import Dense, Activation


class QFunctionNeuralNetwork(implements(IAgent)):
    def __init__(self, player, numFeatures, batchSize=1, gamma=1, alpha=0.1):
        self.player = player
        self.weights = np.ones(numFeatures) * 0.5
        self.s, self.a, self.r = None, None, None
        self.batch = []
        self.batches = 0
        self.batchSize = batchSize
        self.gamma = gamma
        self.alpha = alpha
        self.model = Sequential([
            Dense(100, input_dim=numFeatures),
            Activation('relu'),
            Dense(25),
            Activation('relu'),
            Dense(1),
            Activation('linear')
        ])
        self.model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])

    def getMove(self, state):
        reward = state.getReward(self.player)
        actions = state.getAction(self.player)
        self.updateBatch(state, reward, actions)

        a = actions[argmax([self.Q(state, aP) for aP in actions])]

        self.s = state
        self.a = a

        return self.a

    def getTrainedMove(self, state):
        actions = state.getAction(self.player)
        return actions[argmax([self.Q(state, aP) for aP in actions])]

    def finalize(self, state):
        reward = self.getReward(self.player)
        self.updateBatch(state, reward, None)

    def getInfo(self):
        return

    def Q(self, state, action):
        features = np.array([feature(state, action) for feature in state.getFeatures(self.player)])
        prediction = self.model.predict(np.array([features]))[0]
        return prediction[0]

    def updateBatch(self, state, reward, actions):
        if self.s is not None:
            if actions is None:
                q = (1 - self.alpha) * self.Q(self.s, self.a) + self.alpha * reward
            else:
                q = (1 - self.alpha) * self.Q(self.s, self.a) + self.alpha * (reward + self.gamma * max([self.Q(state, aP) for aP in actions]))

            self.batch.append([np.asarray([feature(self.s, self.a) for feature in state.getFeatures(self.player)]), q])

        if len(self.batch) == self.batchSize:
            X = np.array([b[0] for b in self.batch])
            Y = np.array([b[1] for b in self.batch])

            self.model.fit(X, Y, epochs=30, batch_size=10, validation_split=0.2)

            self.batch = []

        self.batches += 1


def argmax(l):
    return l.index(max(l))
