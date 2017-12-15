from interfaces import IAgent
from interface import implements
import numpy as np


class Random(implements(IAgent)):
    def __init__(self, player):
        """
        :param player: the id of the player this agent plays as
        """
        self.player = player

    def getMove(self, state):
        actions = state.getActions(self.player)
        return actions[np.random.randint(len(actions))]

    def getTrainedMove(self, state):
        return self.getMove(state)

    def finalize(self, state):
        pass

    def getInfo(self):
        return