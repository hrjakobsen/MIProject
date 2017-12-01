from games.hexagon import *

class GreedyHexAgent(object):
    def __init__(self, player):
        self.player = player

    def finalize(self, state, reward, actions):
        pass

    def getMove(self, state: HexagonGame, reward, actions):
        currentBoard = state.board
        bestA = 0
        maxCells = len(getOwnedCells(currentBoard, self.player))
        for a in actions:
            newBoard = makeMove(currentBoard, state.neighbourMap, self.player, a)
            if len(getOwnedCells(newBoard, self.player)) > maxCells:
                bestA = a
                maxCells = len(getOwnedCells(newBoard, self.player))
        return bestA