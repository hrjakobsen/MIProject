from interfaces import IAgent
from interface import implements
import copy
import pickle
import os.path


class QFunctionTabular(implements(IAgent)):
    """
    This class is an implementation of the Q-Learning-Agent
    from Russel & Norvig (2010) p. 844
    """

    def __init__(self, player, Q, N, gamma=1):
        """
        :param player: the id of the player this agent plays as
        :param Q: The Q table to use
        :param N: The N table to use
        :param gamma: discount factor to use
        """
        self.player = player
        self.Q = Q
        self.N = N
        self.gamma = gamma
        self.s = None
        self.hash = None
        self.a = None
        self.r = None

    def getMove(self, state):
        actions = state.getActions(self.player)
        reward = state.getReward(self.player)
        if self.s is not None:
            self._incrementN()
            self._updateQ(state, reward, actions)
        self.s = copy.deepcopy(state)
        self.a = self._argmax(actions)
        self.r = reward
        return self.a

    def getTrainedMove(self, state):
        actions = state.getActions(self.player)
        return self._argmax(actions)

    def finalize(self, state):
        actions = state.getActions(self.player)
        reward = state.getReward(self.player)
        if self.s is None:
            return

        for a in actions:
            self.Q[state.hash(), a] = reward

        self._updateQ(state, reward, actions)

        self.s = None

    def getInfo(self):
        return

    def _argmax(self, actions):
        s = self.s.hash()
        vals = [self._f(self.Q.get((s, a), 0), self.N.get((s, a), 0)) for a in actions]
        return actions.index(vals.index(max(vals)))

    def _f(self, val, num):
        if num < 10:
            return 5
        return val

    def _incrementN(self):
        s = self.s.hash()
        a = self.a
        self.N[s, a] = self.N.get((s, a), 0) + 1

    def _updateQ(self, sP, rP, actions):
        s = self.s.hash()
        sP = sP.hash()
        a = self.a
        self.Q[s, a] = self.Q.get((s, a), 0) + self._alpha() * (
            self.r + self.gamma * (max([self.Q.get((sP, aP), 0) for aP in actions]) - self.Q.get((s, a), 0)))

    def _alpha(self):
        return 1 / (1000 + self.N.get((self.s.hash(), self.a), 0))

    @classmethod
    def load(cls, player):
        if os.path.isfile("saves/P" + str(player) + "_Q.pickle") and os.path.isfile("saves/P" + str(player) + "_N.pickle"):
            return cls(player, loadFromFile("saves/P" + str(player) + "_Q.pickle"),
                       loadFromFile("saves/P" + str(player) + "_N.pickle"))

        print("No save-file was found for player" + str(player) + ". Creating empty agent")
        return cls(player, {}, {})

    def save(self, player):
        if not os.path.exists("saves"):
            os.makedirs("saves")

        saveToFile(self.Q, "saves/P" + str(player) + "_Q.pickle")
        saveToFile(self.N, "saves/P" + str(player) + "_N.pickle")


def saveToFile(dict, fileName):
    with open(fileName, 'wb') as handle:
        pickle.dump(dict, handle, protocol=pickle.HIGHEST_PROTOCOL)


def loadFromFile(fileName):
    with open(fileName, 'rb') as handle:
        return pickle.load(handle)
