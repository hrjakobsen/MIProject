import numpy as np


class BattleshipGame(object):
    def __init__(self, board1=None, board2=None):
        self.playerOneBoard = randomBoard() if board1 is None else board1
        self.playerTwoBoard = randomBoard() if board2 is None else board2

    def getActions(self, player):
        board = self.playerTwoBoard if player == 1 else self.playerOneBoard
        actions = []
        for (x, y), value in np.ndenumerate(board):
            if value != 2:
                actions.append((x, y))
        return actions


    def gameEnded(self):
        return not np.any(self.playerOneBoard == 1) and np.any(self.playerTwoBoard == 1)

    def getFeatures(self, player):
        pass

    def getReward(self, player):
        totalShipSquares = 14
        board = self.playerOneBoard if player == 1 else self.playerTwoBoard
        return totalShipSquares - sum(np.where(board == 1))

    def __deepcopy__(self, _):
        new = BattleshipGame(self.playerOneBoard.copy(), self.playerTwoBoard.copy())
        return new

    def makeMove(self, player, action):
        board = self.playerTwoBoard if player == 1 else self.playerOneBoard
        board[action[0], action[1]] = 2


def randomBoard():
    boardSize = 11
    board = np.zeros((boardSize, boardSize), dtype=int)
    shipLengths = [2, 3, 4, 5]

    for ship in shipLengths:
        horizontal = np.random.rand() < .5
        if horizontal:
            placed = False
            while not placed:
                placeable = True
                row = np.random.randint(0, boardSize)
                col = np.random.randint(0, boardSize - ship)

                for i in range(ship):
                    if board[row, col + i] != 0:
                        placeable = False
                        break

                if placeable:
                    for i in range(ship):
                        board[row, col + i] = 1
                    placed = True
        else:
            placed = False
            while not placed:
                placeable = True
                col = np.random.randint(0, boardSize)
                row = np.random.randint(0, boardSize - ship)

                for i in range(ship):
                    if board[row + i, col] != 0:
                        placeable = False
                        break

                if placeable:
                    for i in range(ship):
                        board[row + i, col] = 1
                    placed = True

    return board