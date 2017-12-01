from games.hexagon import HexagonGame
import copy
import pickle
import os.path

def saveToFile(dict, fileName):
    with open(fileName, 'wb') as handle:
        pickle.dump(dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

def loadFromFile(fileName):
    with open(fileName, 'rb') as handle:
        return pickle.load(handle)

class TabularQLearner(object):
    """
    This class is an implementation of the Q-Learning-Agent
    from Russel & Norvig (2010) p. 844
    """

    def __init__(self, Q, N, gamma=1):
        """
        :param player: the player id of the agent (1 or 2)
        :param Q: The Q table to use
        :param N: The N table to use
        """
        self.Q = Q
        self.N = N
        self.gamma = gamma
        self.s = None
        self.hash = None
        self.a = None
        self.r = None

    def getMove(self, state: HexagonGame, reward, actions):
        """
        Ask the agent what action to take
        :param state: the current game
        :param reward: the current reward
        :param actions: the actions to chose from
        :return: the action to play in this state
        """
        if self.s is not None:
            self._incrementN()
            self._updateQ(state, reward, actions)
        self.s = copy.deepcopy(state)
        self.a = self._argmax(actions)
        self.r = reward
        return self.a

    def _argmax(self, actions):
        """
        :return: the action that results in the highest value from the f-function
        """
        s = self.s.hash()
        vals = [self._f(self.Q.get((s, a), 0), self.N.get((s, a), 0)) for a in actions]
        return actions.index(vals.index(max(vals)))

    def _f(self, val, num):
        """
        This function rewards exploration versus exploitation
        the first 10 times a state is encountered
        :param val: The current Q-estimate
        :param num: The number of times the state has been visited
        :return: val if the state has been visited more than 10 times, otherwise 100
        """
        if num < 20:
            return 2000
        return val

    def _incrementN(self):
        """
        Increments the number of times the agent have seen a state
        :return:
        """
        s = self.s.hash()
        a = self.a
        self.N[s, a] = self.N.get((s, a), 0) + 1

    def _updateQ(self, sP: HexagonGame, rP, actions):
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
            self.r + self.gamma * (max([self.Q.get((sPh, aP), 0) for aP in actions]) - self.Q.get((s, a), 0)))

    def _alpha(self):
        """
        The learning rate parameter is decreasing over time
        :return: the current learning rate parameter for the last state and action
        """
        return 1#100 / (10000 + self.N.get((self.s.hash(), self.a), 0))

    def finalize(self, state, reward, actions):
        if self.s is None:
            return
            
        for a in actions:
            self.Q[state.hash(), a] = reward

        self._updateQ(state, reward, actions)

        self.s = None

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
