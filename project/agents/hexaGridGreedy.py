from games.hexaGrid import getOwnedCells, makeMove
from interfaces import IAgent
from interface import implements


class HexaGridGreedy(implements(IAgent)):
    def __init__(self, player):
        """
        :param player: the id of the player this agent plays as
        """
        self.player = player

    def getMove(self, state):
        currentBoard = state.board
        bestA = 0
        maxCells = len(getOwnedCells(currentBoard, self.player))
        actions = state.getActions(self.player)
        for a in actions:
            newBoard = makeMove(currentBoard, state.neighbourMap, self.player, a)
            if len(getOwnedCells(newBoard, self.player)) > maxCells:
                bestA = a
                maxCells = len(getOwnedCells(newBoard, self.player))
        return bestA

    def getTrainedMove(self, state):
        return self.getMove(state)

    def finalize(self, state):
        pass

    def getInfo(self):
        return