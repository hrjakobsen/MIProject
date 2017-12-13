from interfaces import IAgent
from interface import implements
import numpy as np


class Random(implements(IAgent)):
    def __init__(self, player):
        self.player = player

    def getMove(self, state):
        """
        Ask the agent what action to take
        :param state: the current game
        :param reward: the current reward
        :param actions: the actions to choose from
        :return: the action to play in this state
        """
        actions = state.getActions(self.player)
        return actions[np.random.randint(len(actions))]

    def getTrainedMove(self, state):
        return self.getMove(state)

    def finalize(self, state):
        pass

    def getInfo(self):
        return