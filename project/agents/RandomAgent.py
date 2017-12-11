from games.hexagon import HexagonGame
import numpy as np

class RandomAgent(object):
    def getMove(self, state, reward, actions):
        """
        Ask the agent what action to take
        :param state: the current game
        :param reward: the current reward
        :param actions: the actions to choose from
        :return: the action to play in this state
        """
        return actions[np.random.randint(len(actions))]

    def getTrainedMove(self, state, actions):
        return self.getMove(state, 0, actions)

    def finalize(self, state, reward, action):
        pass