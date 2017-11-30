import numpy as np
import math

WATER = 0
SHIP = 1
WATERHIT = 2
SHIPHIT = 3

class BattleshipGame(object):
    def __init__(self, boardSize=10, ships=[2, 3, 3, 4, 5], board=None):
        self.boardSize = boardSize
        self.actions = None
        self.board, self.ships = randomBoard(boardSize, ships) if board is None else board
        self.hits = []
        self.numHits = 0
        self.misses = []
        self.numFeatures = None

    def getActions(self, player):
        if self.actions is None:
            actions = []
            for (x, y), value in np.ndenumerate(self.board):
                if value < WATERHIT:
                    actions.append((x, y))
            self.actions = actions

        return self.actions

    def gameEnded(self):
        return not np.any(self.board == SHIP)

    def getNumFeatures(self):
        if self.numFeatures is None:
            self.numFeatures = len(calculateFeatures(self, (0, 0), 1))

        return self.numFeatures

    def calculateFeatures(self, state, action, player):
        return calculateFeatures(state, action, player)

    def getReward(self, player):
        if self.gameEnded():
            numMoves = len(np.where(self.board > 1)[0])
            return (self.numHits * 20) - numMoves

        return 0

    def __deepcopy__(self, _):
        new = BattleshipGame(self.boardSize, [], self.board)
        new.actions = self.actions
        new.ships = self.ships
        new.hits = self.hits
        new.numHits = self.numHits
        new.misses = self.misses
        new.numFeatures = self.numFeatures
        return new

    def makeMove(self, player, action):
        self.board[action[0], action[1]] += 2
        self.actions.remove(action)

        if self.board[action[0], action[1]] == WATERHIT:
            self.misses.append(action)

        if self.board[action[0], action[1]] == SHIPHIT:
            self.numHits += 1
            self.hits.append(action)

            for ship in self.ships:
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

def calculateFeatures(state, action, player):
    results = np.array([
        1,
        distanceToHit(state, action, player),
        distanceToMiss(state, action, player),
        hitsOnALine(state, action, player)
    ])

    return results

def distanceToHit(state, action, player):
    minDist = state.boardSize * 2
    for hit in state.hits:
        # Manhattan distance
        tempDist = abs(action[0] - hit[0]) + abs(action[1] - hit[1])
        if tempDist < minDist:
            minDist = tempDist

    return minDist

def distanceToMiss(state, action, player):
    minDist = state.boardSize * 2
    for miss in state.misses:
        # Manhattan distance
        tempDist = abs(action[0] - miss[0]) + abs(action[1] - miss[1])
        if tempDist < minDist:
            minDist = tempDist

    return minDist

def hitsOnALine(state, action, player):
    #Is this action on a line, looking at previous hits?

    #print("---")
    #print(action)
    #print(state.hits)

    if len(state.hits) >= 2:
        for hit in state.hits:
            for otherHit in state.hits:
                #print("Checking {0} against {1}".format(hit, otherHit))
                dRow = hit[0] - otherHit[0]
                dCol = hit[1] - otherHit[1]

                if dRow == 0 and dCol != 0:
                    newAction1 = (hit[0], otherHit[1] + 1)
                    newAction2 = (hit[0], hit[1] - 1)
                elif dRow != 0 and dCol == 0:
                    newAction1 = (hit[0] + 1, hit[1])
                    newAction2 = (otherHit[0] - 1, hit[1])
                else:
                    continue

                #print("New actions:")
                #print(newAction1)
                #print(newAction2)
                if newAction1 == action or newAction2 == action:
                    #print(1)
                    #print("---")
                    return 1

    #print(0)
    #print("---")
    return 0

