from games.HexagonGame import HexagonGame
import numpy as np

class RandomAgent(object):
    def __init__(self, actions):
        self.actions = actions

    def getMove(self, state: HexagonGame, reward):
        """
        Ask the agent what action to take
        :param state: the current game
        :param reward: the current reward
        :return: the action to play in this state
        """
        return self.actions[np.random.randint(len(self.actions))]


    def finalize(self, state, reward):
        pass
