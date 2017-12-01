from games.hexagon import HexagonGame
import math
import numpy as np

class HexFALearner(object):
    def __init__(self, player, width, height, batchSize, weights=None):
        self.player = player
        self.width = width
        self.height = height
        self.batchSize = batchSize
        self.numWeights = len(weights) if weights is not None else 7 * ((height + 1) * width - (width // 2)) + 1
        self.weights = np.array(weights) if weights is not None else np.ones(self.numWeights)
        self.s = None
        self.a = None
        self.actions = [0, 1, 2, 3, 4]
        self.gamma = 1
        self.batch = []
        self.numBatches = 0


    def getMove(self, state: HexagonGame, reward):
        """
        Ask the agent what action to take
        :param state: the current game
        :param reward: the current reward
        :return: the action to play in this state
        """
        if len(self.batch) >= self.batchSize:
            self._updateWeights()
            self.batch = []
            self.numBatches += 1

        maxQ = self._calculateQ(state, 0)
        bestAction = 0

        for action in range(1, len(self.actions)):
            Q = self._calculateQ(state, action)
            if Q > maxQ:
                maxQ = Q
                bestAction = action

        if self.s is not None:
            q = (1 - self._alpha()) * self._calculateQ(self.s, self.a) + self._alpha() * (reward + self.gamma * maxQ)
            self.batch.append({"state": self.s, "action": self.a, "Q": q})

        self.s = state
        self.a = bestAction

        return bestAction


    def _updateWeights(self):
        newWeights = self.weights.copy()
        differences = []

        for batch in self.batch:
            differences.append(-1 * (batch['Q'] - self._calculateQ(batch['state'], batch['action'])))

        for j in range(self.numWeights):
            batchSum = 0
            for i, batch in enumerate(self.batch):
                batchSum += differences[i] * self._feature(j, batch['state'], batch['action'])
            newWeights[j] -= (self._alpha() * 2 * batchSum) / len(self.batch)

        self.weights = newWeights

    def _feature(self, featureNumber, state: HexagonGame, action):
        if featureNumber == 0:
            return 1

        correctedFeatureNumber = featureNumber - 1
        # 0-4: colors, 5, 6: P1 and P2
        code = correctedFeatureNumber % 7

        cellNumber = correctedFeatureNumber // 7
        height = self.height + 1
        x = math.floor(cellNumber / (height - 0.5))
        yP = cellNumber % (height * 2 - 1)
        y = yP % height + yP // height

        if code < 5:
            return state.board[y, x] == code

        if code == 5:
            return 5 <= state.board[y, x] < 10

        if code == 6:
            return state.board[y, x] >= 10

    def _calculateQ(self, state, action):
        features = np.empty(self.numWeights)
        for x in range(self.numWeights):
            features[x] = (self._feature(x, state, action))

        return np.sum(self.weights * features)


    def _alpha(self):
        """
        The learning rate parameter is decreasing over time
        :return: the current learning rate parameter for the last state and action
        """
        return 60 / (60 + self.numBatches)


    def finalize(self, state, reward):
        if len(self.batch) >= self.batchSize:
            self._updateWeights()
            self.batch = []
            self.numBatches += 1

        maxQ = self._calculateQ(state, 0)

        for action in range(1, len(self.actions)):
            Q = self._calculateQ(state, action)
            if Q > maxQ:
                maxQ = Q

        if self.s is not None:
            q = (1 - self._alpha()) * self._calculateQ(self.s, self.a) + self._alpha() * (reward + self.gamma * maxQ)
            self.batch.append({"state": self.s, "action": self.a, "Q": reward})

        self.s = None
        self.a = None