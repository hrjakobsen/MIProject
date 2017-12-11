import numpy as np
import copy

WATER = 0
SHIP = 1
WATERHIT = 2
SHIPHIT = 3

class BattleshipGame(object):
    def __init__(self, boardSize=10, ships=[2, 3, 3, 4, 5]):
        self.p1Game = _BattleshipSingleGame(boardSize, ships)
        self.p2Game = _BattleshipSingleGame(boardSize, ships)
        self.numFeatures = None

    def getActions(self, player):
        # Actions available on the opponent's board
        return self.p2Game.getActions() if player == 1 else self.p1Game.getActions()

    def gameEnded(self):
        return self.p1Game.gameEnded() or self.p2Game.gameEnded()

    def calculateFeatures(self, state, action, player):
        return self.p2Game.calculateFeatures(state, action) if player == 1 else self.p1Game.calculateFeatures(state, action)

    def getReward(self, player):
        return self.p2Game.getReward() if player == 1 else self.p1Game.getReward()

    def makeMove(self, player, action):
        return self.p2Game.makeMove(action) if player == 1 else self.p1Game.makeMove(action)


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

    def __deepcopy__(self, _):
        new = BattleshipGame(self.boardSize, self.ships)
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

    def calculateFeatures(self, state, action):
        return calculateFeatures(state, action)

    def getReward(self):
        if self.gameEnded():
            numMoves = len(np.where(self.board > 1)[0])
            return (self.numHits * 20) - numMoves

        return 0

    def makeMove(self, action):
        self.board[action[0], action[1]] += 2
        self.actions.remove(action)

        if self.board[action[0], action[1]] == WATERHIT:
            self.misses.append(action)

        if self.board[action[0], action[1]] == SHIPHIT:
            self.numHits += 1
            self.hits.append(action)

        return
        for ship in self.shipStatus:
            if (action, True) in ship:
                ship[ship.index((action, True))] = (action, False)
                shipSunk = True
                for cell in ship:
                    if cell[1]:
                        shipSunk = False
                        break

                if shipSunk:
                    for cell in ship:
                        self.hits.remove(cell[0])

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


def calculateFeatures(state, action):
    results = np.array([
        1,
        distanceToHitOrMissSquare(state, action, state.hits),
        distanceToHitOrMissSquare(state, action, state.misses),
        hitsOnALine(state, action)
    ])

    return results

def distanceToHitOrMissSquare(state, action, squares):
    minDist = state.boardSize * 2
    for square in squares:
        # Manhattan distance
        tempDist = abs(action[0] - square[0]) + abs(action[1] - square[1])
        if tempDist < minDist:
            minDist = tempDist

    return (minDist - 1) / (state.boardSize * 2 - 1)

def hitsOnALine(state, action):
    for hit in state.hits:
        if action[0] == hit[0] or action[1] == hit[1]:
            return 1

    return 0
