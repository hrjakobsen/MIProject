import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation

class NeuralNetworkAgent(object):
    def __init__(self, player, numFeatures, actions, batchSize=1, gamma=1):
        self.player = player
        self.weights = np.ones(numFeatures) * 0.5
        self.s, self.a, self.r = None, None, None
        self.actions = actions
        self.batch = []
        self.batches = 0
        self.batchSize = batchSize
        self.gamma = gamma
        self.model = Sequential([
            Dense(100, input_dim=numFeatures),
            Activation('relu'),
            Dense(25),
            Activation('relu'),
            Dense(1),
            Activation('linear')
        ])
        self.model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])

    def Q(self, state, action):
        features = np.array([feature(state, action) for feature in state.getFeatures(self.player)])
        prediction = self.model.predict(np.array([features]))[0]
        return prediction[0]

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
            self.batch.append([np.asarray([feature(self.s, self.a) for feature in state.getFeatures(self.player)]), q])

        if len(self.batch) == self.batchSize:
            X = np.array([b[0] for b in self.batch])
            Y = np.array([b[1] for b in self.batch])

            self.model.fit(X, Y, epochs=30, batch_size=10, validation_split=0.2)

            self.batch = []

        self.batches += 1

    def finalize(self, state, reward):
        self.updateBatch(state, reward)

    def alpha(self):
        return 0.005

def argmax(l):
    return l.index(max(l))
