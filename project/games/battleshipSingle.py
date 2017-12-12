from Interfaces import IGame
from interface import implements
import numpy as np
import copy


class BattleshipGame(implements(IGame)):
    def __init__(self, boardSize=10, ships=[2, 3, 3, 4, 5]):
        self.boardSize = boardSize
        self.ships = ships
        self.p1Game = _BattleshipSingleGame(boardSize, ships)
        self.p2Game = _BattleshipSingleGame(boardSize, ships)
        self.numFeatures = None

    def __deepcopy__(self, _):
        new = BattleshipGame(self.boardSize, self.ships)
        new.p1Game = copy.deepcopy(self.p1Game)
        new.p2Game = copy.deepcopy(self.p2Game)
        new.numFeatures = self.numFeatures
        return new

    def getActions(self, player):
        # Actions available on the opponent's board
        return self.p2Game.getActions() if player == 1 else self.p1Game.getActions()

    def getNumFeatures(self):
        if self.numFeatures is None:
            self.numFeatures = len(self.p1Game.calculateFeatures((0, 0)))
        return self.numFeatures

    def getFeatures(self, player, action):
        return self.p2Game.calculateFeatures(action) if player == 1 else self.p1Game.calculateFeatures(action)

    def gameEnded(self):
        return self.p1Game.gameEnded() or self.p2Game.gameEnded()

    def getReward(self, player):
        return self.p2Game.getReward() if player == 1 else self.p1Game.getReward()

    def makeMove(self, player, action):
        if player == 1:
            self.p2Game.makeMove(action)
        else:
            self.p1Game.makeMove(action)


WATER = 0
SHIP = 1
WATERHIT = 2
SHIPHIT = 3


class _BattleshipSingleGame(object):
    def __init__(self, boardSize=10, ships=[2, 3, 3, 4, 5]):
        self.boardSize = boardSize
        self.actions = None
        self.ships = ships
        self.board, self.shipStatus = randomBoard(boardSize, ships)
        self.hits = []
        self.numHits = 0
        self.misses = []
        self.numFeatures = None
        self.numMoves = 0
        self.removedShipSquares = []

    def __deepcopy__(self, _):
        new = _BattleshipSingleGame(self.boardSize, self.ships)
        new.board = self.board.copy()
        new.shipStatus = copy.deepcopy(self.shipStatus)
        new.hits = self.hits.copy()
        new.numHits = self.numHits
        new.misses = self.misses.copy()
        new.numFeatures = self.numFeatures
        return new

    def getActions(self):
        if self.actions is None:
            actions = []
            for (x, y), value in np.ndenumerate(self.board):
                if value < WATERHIT:
                    actions.append((x, y))
            self.actions = actions

        return self.actions

    def gameEnded(self):
        return not np.any(self.board == SHIP)

    def calculateFeatures(self, action):
        results = np.array([
            1,
            distanceToSquares(self, action, self.misses),
            distanceToSquares(self, action, self.hits),
            hitsOnALine(self, action),
            #chanceOfHittingShip(state, action)
        ])

        return results

    def getReward(self):
        if self.gameEnded():
            return (self.numHits * 20) * (self.boardSize / self.numMoves)

        return 0

    def makeMove(self, action):
        self.board[action[0], action[1]] += 2
        self.actions.remove(action)
        self.numMoves += 1

        if self.board[action[0], action[1]] == WATERHIT:
            self.misses.append(action)

        if self.board[action[0], action[1]] == SHIPHIT:
            self.numHits += 1
            self.hits.append(action)
            self.numMoves -= 1


def randomBoard(boardSize, ships):
    board = np.zeros((boardSize, boardSize), dtype=int)
    shipsList = []

    for ship in ships:
        shipsList.append([])
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
                        shipsList[-1].append(((row, col + i), True))
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
                        shipsList[-1].append(((row + i, col), True))
                    placed = True

    return board, shipsList


def distanceToSquares(state, action, squares):
    minDist = state.boardSize * 2
    for square in squares:
        # Manhattan distance
        tempDist = abs(action[0] - square[0]) + abs(action[1] - square[1])
        if tempDist < minDist:
            minDist = tempDist

    return (minDist - 1) / (state.boardSize * 2 - 1)


# 610
def hitsOnALine(state, action):
    for hit in state.hits:
        if action[0] == hit[0] or action[1] == hit[1]:
            return 1

    return 0

"""
# 594
def hitsOnALine(state, action):
    hitBoard = {}
    board = state.board

    for hit in state.hits:
        hx, hy = hit
        # look left
        for dx in range(hx, 0, -1):
            if board[dx, hy] != SHIPHIT:
                dist = abs(dx-hx)
                hitBoard[dx, hy] = hitBoard.get((dx, hy), 0) + 1/dist
                break
        # look right
        for dx in range(hx, state.boardSize, 1):
            if board[dx, hy] != SHIPHIT:
                dist = abs(dx-hx)
                hitBoard[dx, hy] = hitBoard.get((dx, hy), 0) + 1/dist
                break
        for dy in range(hy, 0, -1):
            if board[hx, dy] != SHIPHIT:
                dist = abs(dy-hy)
                hitBoard[hx, dy] = hitBoard.get((hx, dy), 0) + 1/dist
                break
        for dy in range(hy, state.boardSize, 1):
            if board[hx, dy] != SHIPHIT:
                dist = abs(dy-hy)
                hitBoard[hx, dy] = hitBoard.get((hx, dy), 0) + 1/dist
                break

    actionValue = hitBoard.get(action, 0)
    return actionValue
"""

"""
# 573
def hitsOnALine(state, action ,player):
    hitBoard = {}
    board = state.board

    for hit in state.hits:
        hx, hy = hit
        # look left
        for dx in range(hx, 0, -1):
            if board[dx, hy] != SHIPHIT:
                dist = abs(dx-hx)
                hitBoard[dx, hy] = hitBoard.get((dx, hy), 0) + 1/dist
        # look right
        for dx in range(hx, state.boardSize, 1):
            if board[dx, hy] != SHIPHIT:
                dist = abs(dx-hx)
                hitBoard[dx, hy] = hitBoard.get((dx, hy), 0) + 1/dist
        for dy in range(hy, 0, -1):
            if board[hx, dy] != SHIPHIT:
                dist = abs(dy-hy)
                hitBoard[hx, dy] = hitBoard.get((hx, dy), 0) + 1/dist
        for dy in range(hy, state.boardSize, 1):
            if board[hx, dy] != SHIPHIT:
                dist = abs(dy-hy)
                hitBoard[hx, dy] = hitBoard.get((hx, dy), 0) + 1/dist

    actionValue = hitBoard.get(action, 0)
    return actionValue
"""

""" 
# 528 
def hitsOnALine(state, action, player):
    hitBoard = {}
    board = state.board
    for hit in state.hits:
        hx, hy = hit
        for dx in range(hx, 0, -1):
            if board[dx, hy] != SHIPHIT:
                hitBoard[dx, hy] = hitBoard.get((dx, hy), 0) + 1
        for dx in range(hx, state.boardSize, 1):
            if board[dx, hy] != SHIPHIT:
                hitBoard[dx, hy] = hitBoard.get((dx, hy), 0) + 1
        for dy in range(hy, 0, -1):
            if board[hx, dy] != SHIPHIT:
                hitBoard[hx, dy] = hitBoard.get((hx, dy), 0) + 1
        for dy in range(hy, state.boardSize, 1):
            if board[hx, dy] != SHIPHIT:
                hitBoard[hx, dy] = hitBoard.get((hx, dy), 0) + 1

    return hitBoard.get(action, 0)
    """

def chanceOfHittingShip(state, action):
    sizeOfShip = 5 # todo fix this?
    count = 0
    actionX, actionY = action
    for xOffset in range(0, sizeOfShip + 1):
        highX = actionX - xOffset + sizeOfShip
        lowX  = actionX - xOffset
        if 0 <= highX < state.boardSize and 0 <= lowX < state.boardSize:
            count += 1

    for yOffset in range(0, sizeOfShip + 1):
            highY = actionY - yOffset + sizeOfShip
            lowY  = actionY - yOffset
            if 0 <= highY < state.boardSize and 0 <= lowY < state.boardSize:
                count += 1
    return count
