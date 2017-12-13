from interfaces import IAgent
from interface import implements
import numpy as np


class BattleShipHuntAndTarget(implements(IAgent)):
    def __init__(self, player, boardSize):
        self.player = player
        self.boardSize = boardSize

    def getMove(self, state):
        """
        Ask the agent what action to take
        :param state: the current game
        :param reward: the current reward
        :param actions: the actions to choose from
        :return: the action to play in this state
        """
        actions = state.getActions(self.player)
        hits = state.p2Game.hits if self.player == 1 else state.p1Game.hits
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
                    if otherHit[0] == self.boardSize-1:
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
                    if otherHit[1] == self.boardSize-1:
                        action = (hit[0], hit[1]-1)
                    else:
                        action = (otherHit[0], otherHit[1]+1)

                for availableAction in actions:
                    if action == availableAction:
                        return action

        #if we don't have two hits next to each other, check if we at least got 1 hit, then shoot next to it
        if hits:
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

    def getTrainedMove(self, state):
        return self.getMove(state)

    def finalize(self, state):
        pass

    def getInfo(self):
        return
