from games.HexagonGame import *

class GreedyHexAgent(object):
    def __init__(self, player):
        self.player = player
        self.actions = [0, 1, 2, 3, 4]

    def finalize(self, state, reward):
        pass

    def getMove(self, state: HexagonGame, reward):
        currentBoard = state.board
        bestA = 0
        maxCells = len(getOwnedCells(currentBoard, self.player))
        for a in self.actions:
            newBoard = makeMove(currentBoard, state.neighbourMap, self.player, a)
            if len(getOwnedCells(newBoard, self.player)) > maxCells:
                bestA = a
        return bestA