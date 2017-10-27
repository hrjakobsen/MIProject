from games.HexagonGame import HexagonGame
import copy
import json
import ast


def saveToFile(dict, fileName):
    with open(fileName, 'w') as fp:
        json.dump({str(k): v for k, v in dict.items()}, fp)


def loadFromFile(fileName):
    with open(fileName, 'r') as fp:
        return {ast.literal_eval(k): v for k, v in json.load(fp).items()}


class HexLearner(object):
    """
    This class is an implementation of the Q-Learning-Agent
    from Russel & Norvig (2010) p. 844
    """

    def __init__(self, player, Q={}, N={}):
        """
        :param player: the player id of the agent (1 or 2)
        :param Q: The Q table to use
        :param N: The N table to use
        """
        self.Q = Q
        self.N = N
        self.s = None
        self.hash = None
        self.a = None
        self.r = None
        self.player = player
        self.actions = [0, 1, 2, 3, 4]

    def getMove(self, state: HexagonGame, reward):
        """
        Ask the agent what action to take
        :param state: the current game
        :param reward: the current reward
        :return: the action to play in this state
        """
        if self.s is not None:
            self._incrementN()
            self._updateQ(state, reward)
        self.s = copy.deepcopy(state)
        self.a = self._argmax()
        self.r = reward
        return self.a

    def _argmax(self):
        """
        :return: the action that results in the highest value from the f-function
        """
        s = self.s.hash()
        vals = [self._f(self.Q.get((s, a), 0), self.N.get((s, a), 0)) for a in self.actions]
        return self.actions.index(vals.index(max(vals)))

    def _f(self, val, num):
        """
        This function rewards exploration versus exploitation
        the first 10 times a state is encountered
        :param val: The current Q-estimate
        :param num: The number of times the state has been visited
        :return: val if the state has been visited more than 10 times, otherwise 100
        """
        if num < 10:
            return 100
        return val

    def _incrementN(self):
        """
        Increments the number of times the agent have seen a state
        :return:
        """
        s = self.s.hash()
        a = self.a
        self.N[s, a] = self.N.get((s, a), 0) + 1

    def _updateQ(self, sP: HexagonGame, rP):
        """
        Do the update rule for the Q-table
        :param sP: The current state
        :param rP: The current reward
        :return:
        """
        s = self.s.hash()
        sPh = sP.hash()
        a = self.a
        self.Q[s, a] = self.Q.get((s, a), 0) + self._alpha() * (
            self.r + max([self.Q.get((sPh, aP), 0) for aP in self.actions]) - self.Q.get((s, a), 0))

    def _alpha(self):
        """
        The learning rate parameter is decreasing over time
        :return: the current learning rate parameter for the last state and action
        """
        return 60 / (59 + self.N.get((self.s.hash(), self.a), 0))

    def finalize(self, state, reward):
        if self.s is None:
            return
        for a in self.actions:
            self.Q[state.hash(), a] = reward

        self._updateQ(state, reward)

    @classmethod
    def load(cls, player):
        return cls(player, loadFromFile("saves/P" + str(player) + "_Q.json"),
                   loadFromFile("saves/P" + str(player) + "_N.json"))

    def save(self):
        saveToFile(self.Q, "saves/P" + str(self.player) + "_Q.json")
        saveToFile(self.N, "saves/P" + str(self.player) + "_N.json")
