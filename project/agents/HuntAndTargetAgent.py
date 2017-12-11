#from games.hexagon import HexagonGame
from games.battleshipSingle import BattleshipGame
import numpy as np

class HuntAndTargetAgent(object):
    def __init__(self, boardsize):
        self.boardsize = boardsize

    def getMove(self, state: BattleshipGame, reward, actions):
        hits = state.hits
        #if hits != []:
        #    for shipSquare in state.removedShipSquares:
        #        hits.remove(shipSquare)
        #        state.removedShipSquares.remove(shipSquare)
        """
        Ask the agent what action to take
        :param state: the current game
        :param reward: the current reward
        :param actions: the actions to choose from
        :return: the action to play in this state
        """
        #print(state.hits)
        for hit in hits:
            for otherHit in hits:
                action = None
                if hit == otherHit:
                    continue

                #we check for horizontally placed battleship
                if otherHit == (hit[0]-1, hit[1]):
                    if otherHit[0] == 0:
                        action = (hit[0]+1, hit[1])
                    else:
                        action = (otherHit[0]-1, otherHit[1])
                if otherHit == (hit[0]+1, hit[1]):
                    if otherHit[0] == self.boardsize-1:
                        action = (hit[0]-1, hit[1])
                    else:
                        action = (otherHit[0]+1, otherHit[1])

                #we check for vertically placed battleship
                if otherHit == (hit[0], hit[1]-1):
                    if otherHit[1] == 0:
                        action = (hit[0], hit[1]+1)
                    else:
                        action = (otherHit[0], otherHit[1]-1)
                if otherHit == (hit[0], hit[1]+1):
                    if otherHit[1] == self.boardsize-1:
                        action = (hit[0], hit[1]-1)
                    else:
                        action = (otherHit[0], otherHit[1]+1)

                for availableAction in actions:
                    if action == availableAction:
                        return action

        #if we don't have two hits next to each other, check if we at least got 1 hit, then shoot next to it
        if hits != []:
            hit = hits[0]
            for availableAction in actions:
                if availableAction == (hit[0]-1, hit[1]):
                    return availableAction
                if availableAction == (hit[0]+1, hit[1]):
                    return availableAction
                if availableAction == (hit[0], hit[1]-1):
                    return availableAction
                if availableAction == (hit[0], hit[1]+1):
                    return availableAction

        return actions[np.random.randint(len(actions))]

    def getTrainedMove(self, state, actions):
        return self.getMove(state, 0, actions)

    def finalize(self, state, reward, action):
        pass