import numpy as np

WATER = 0
SHIP = 1
WATERHIT = 2
SHIPHIT = 3

boardSize = 11


class BattleshipGame(object):
    def __init__(self, board1=None, board2=None):
        self.playerOneBoard = randomBoard() if board1 is None else board1
        self.playerTwoBoard = randomBoard() if board2 is None else board2

    def getActions(self, player):
        board = self.playerTwoBoard if player == 1 else self.playerOneBoard
        actions = []
        for (x, y), value in np.ndenumerate(board):
            if value != WATERHIT and value != SHIPHIT:
                actions.append((x, y))
        return actions

    def gameEnded(self):
        return not np.any(self.playerOneBoard == 1) and np.any(self.playerTwoBoard == 1)

    def getFeatures(self, player):
        features = [
            lambda s, a: 1
        ]

        for rowa in range(boardSize):
            for cola in range(boardSize):
                for row in range(boardSize):
                    for col in range(boardSize):
                        expectedA = (rowa, cola)
                        features += [
                            lambda s, a, row=row, col=col, expectedA=expectedA: isNotHit(row, col, expectedA, player, s, a),
                            lambda s, a, row=row, col=col, expectedA=expectedA: hitShip(row, col, expectedA, player, s, a),
                            lambda s, a, row=row, col=col, expectedA=expectedA: hitWater(row, col, expectedA, player, s, a)
                        ]

        return features

    def getReward(self, player):
        totalShipSquares = 14
        board = self.playerOneBoard if player == 1 else self.playerTwoBoard
        return totalShipSquares - sum(np.where(board == 1))

    def __deepcopy__(self, _):
        new = BattleshipGame(self.playerOneBoard.copy(), self.playerTwoBoard.copy())
        return new

    def makeMove(self, player, action):
        board = self.playerTwoBoard if player == 1 else self.playerOneBoard
        board[action[0], action[1]] = WATERHIT if board[action[0], action[1]] == WATER else SHIPHIT


def getOppositeSide(player, s: BattleshipGame):
    return s.playerTwoBoard if player == 1 else s.playerOneBoard


def isNotHit(row, col, expectedA, player, s, a):
    if expectedA != a: return 0
    cell = getOppositeSide(player, s)[row, col]
    return 1 if cell == WATER or cell == SHIP else 0


def hitShip(row, col, expectedA, player, s, a):
    if expectedA != a: return 0
    return 1 if getOppositeSide(player, s)[row, col] == SHIPHIT else 0


def hitWater(row, col, expectedA, player, s, a):
    if expectedA != a: return 0
    return 1 if getOppositeSide(player, s)[row, col] == WATERHIT else 0


def randomBoard():
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
                    if board[row, col + i] != WATER:
                        placeable = False
                        break

                if placeable:
                    for i in range(ship):
                        board[row, col + i] = SHIP
                    placed = True
        else:
            placed = False
            while not placed:
                placeable = True
                col = np.random.randint(0, boardSize)
                row = np.random.randint(0, boardSize - ship)

                for i in range(ship):
                    if board[row + i, col] != WATER:
                        placeable = False
                        break

                if placeable:
                    for i in range(ship):
                        board[row + i, col] = SHIP
                    placed = True

    return board
