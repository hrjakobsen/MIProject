from interfaces import IAgent
from interface import implements
from keras import *
from keras.layers import *
from copy import deepcopy

class QFunctionNN(implements(IAgent)):

    def __init__(self, player, numFeatures):
        self.player = player
        self.numFeatures = numFeatures
        self.model = Sequential()
        self.model.add(Dense(10, input_dim=numFeatures, activation="relu"))
        self.model.add(Dense(10, activation="relu"))
        self.model.add(Dense(10, activation="relu"))
        self.model.add(Dense(10, activation="relu"))
        self.model.add(Dense(10, activation="relu"))
        self.model.add(Dense(10, activation="relu"))
        self.model.add(Dense(10, activation="relu"))
        self.model.add(Dense(10, activation="relu"))
        self.model.add(Dense(1, activation="linear"))
        optimizer = optimizers.sgd(lr=0.01)
        self.model.compile(loss='mean_squared_error', optimizer=optimizer)
        self.a = None
        self.s = None
        self.moves = []

    def Q(self, state, action):
        features = np.array(state.getFeatures(self.player, action))
        output = self.model.predict(np.reshape(features, (1, self.numFeatures)))
        return output[0]

    def getMove(self, state):
        reward = state.getReward(self.player)
        actions = state.getActions(self.player)

        if self.s is not None:
            self.updateNetwork(state, actions, reward)

        self.a = self.getTrainedMove(state)
        self.s = deepcopy(state)
        self.moves.append(self.s)

        return self.a

    def updateNetwork(self, state, actions, reward):
        currentEstimate = self.Q(self.s, self.a)
        bestQ = max([self.Q(state, a) for a in actions])
        learning_rate = 0.7
        gamma = 1
        Y = (1-learning_rate)*currentEstimate + learning_rate * (reward + gamma * bestQ)
        inputFeatures = np.array(self.s.getFeatures(self.player, self.a))
        self.model.fit(np.reshape(inputFeatures, (1, self.numFeatures)), np.array([Y]), verbose=0)

    def getTrainedMove(self, state):
        actions = state.getActions(self.player)
        highestQ, highestIndex = self.Q(state, actions[0]), 0
        for i in range(1, len(actions)):
            newQ = self.Q(state, actions[i])
            if newQ > highestQ:
                highestQ = newQ
                highestIndex = i
        return actions[highestIndex]

    def finalize(self, state):
        if self.s is not None:
            inputFeatures = np.array(self.s.getFeatures(self.player, self.a))
            self.model.fit(np.reshape(inputFeatures, (1, self.numFeatures)), np.array([state.getReward(self.player)]), verbose=0)
        self.a, self.s = None, None

    def getInfo(self):
        pass