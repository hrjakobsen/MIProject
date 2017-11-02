from games.HexagonGame import HexagonGame

class HexFALearner(object):
    def __init__(self, player, width, height, batchSize, weights=None):
        self.player = player
        self.width = width
        self.height = height
        self.batchSize = batchSize
        self.weights = weights if weights is not None else [0] * (7 * ((height + 1) * width - (width // 2)) + 1)
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

        for action in self.actions:
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
            differences.append(batch['Q'] - self._calculateQ(batch['state'], batch['action']))

        for j, _ in enumerate(self.weights):
            batchSum = 0
            for i, batch in enumerate(self.batch):
                batchSum += 2 * (differences[i]) * -1 * (self._feature(j, batch['state'], batch['action']))

            newWeights[j] -= (1 / len(self.batch)) * self._alpha() * batchSum

        self.weights = newWeights


    def _feature(self, featureNumber, state: HexagonGame, action):
        if featureNumber == 0:
            return 1

        correctedFeatureNumber = featureNumber - 1
        # 0-4: colors, 5, 6: P1 and P2
        code = correctedFeatureNumber % 7

        # Each even column is one longer than height
        x = 0
        y = correctedFeatureNumber // 7
        while y >= self.height + 1 - x % 2:
            y -= self.height + 1 - x % 2
            x += 1

        # Skip top element in each odd column
        y += x % 2

        if code < 5:
            temp = state.board[y, x] == code
            return temp

        if code == 5:
            temp = 5 <= state.board[y, x] < 10
            return temp

        if code == 6:
            temp = state.board[y, x] >= 10
            return temp

    def _calculateQ(self, state, action):
        total = 0

        for x in range(len(self.weights)):
            total += self.weights[x] * self._feature(x, state, action)

        return total


    def _alpha(self):
        """
        The learning rate parameter is decreasing over time
        :return: the current learning rate parameter for the last state and action
        """
        return 60 / (60 + self.numBatches)


    def finalize(self, state, reward):
        maxQ = -9999999

        for action in self.actions:
            Q = self._calculateQ(state, action)
            if Q > maxQ:
                maxQ = Q

        if self.s is not None:
            q = (1 - self._alpha()) * self._calculateQ(self.s, self.a) + self._alpha() * (reward + self.gamma * maxQ)
            self.batch.append({"state": self.s, "action": self.a, "Q": q})
