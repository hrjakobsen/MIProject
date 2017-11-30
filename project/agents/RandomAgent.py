from games.HexagonGame import HexagonGame
import numpy as np

class RandomAgent(object):

    def getMove(self, state, reward, actions):
        """
        Ask the agent what action to take
        :param state: the current game
        :param reward: the current reward
        :return: the action to play in this state
        """
        return actions[np.random.randint(len(actions))]


    def finalize(self, state, reward):
        pass
