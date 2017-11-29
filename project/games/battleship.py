import numpy as np
import math

WATER = 0
SHIP = 1
WATERHIT = 2
SHIPHIT = 3

boardSize = 11


class BattleshipGame(object):
    def __init__(self, boardSize=10, ships=[2, 3, 3, 4, 5], board1=None, board2=None):
        self.boardSize = boardSize
        self.ships = ships
        self.playerOneBoard, self.playerOneShips = randomBoard(boardSize, ships) if board1 is None else board1
        self.playerTwoBoard, self.playerTwoShips = randomBoard(boardSize, ships) if board2 is None else board2
        self.playerOneBoardHits = []
        self.playerTwoBoardHits = []
        self.playerOneBoardMisses = []
        self.playerTwoBoardMisses = []
        self.numFeatures = None

    def getActions(self, player):
        board = self.playerTwoBoard if player == 1 else self.playerOneBoard
        actions = []
        for (x, y), value in np.ndenumerate(board):
            if value < 2:
                actions.append((x, y))
        return actions

    def gameEnded(self):
        return not (np.any(self.playerOneBoard == 1) and np.any(self.playerTwoBoard == 1))

    def getNumFeatures(self):
        if self.numFeatures is None:
            self.numFeatures = len(calculateFeatures(self, (0, 0), 1))

        return self.numFeatures

    def calculateFeatures(self, state, action, player):
        return calculateFeatures(state, action, player)

    def getReward(self, player):
        if self.gameEnded():
            numHits = len(self.playerTwoBoardHits) if player == 1 else len(self.playerOneBoardHits)
            board = self.playerTwoBoard if player == 1 else self.playerOneBoard
            numMoves = len(np.where(board > 1)[0])
            return (numHits * 20) - numMoves

        return 0

    def __deepcopy__(self, _):
        new = BattleshipGame(self.boardSize, self.ships, self.playerOneBoard.copy(), self.playerTwoBoard.copy())
        new.playerOneBoardHits = self.playerOneBoardHits
        new.playerTwoBoardHits = self.playerTwoBoardHits
        new.numFeatures = self.numFeatures
        return new

    def makeMove(self, player, action):
        board = self.playerTwoBoard if player == 1 else self.playerOneBoard
        boardHits = self.playerTwoBoardHits if player == 1 else self.playerOneBoardHits
        boardMisses = self.playerTwoBoardMisses if player == 1 else self.playerOneBoardMisses
        ships = self.playerTwoShips if player == 1 else self.playerOneShips

        board[action[0], action[1]] += 2

        if board[action[0], action[1]] == 2:
            boardMisses.append(action)

        if board[action[0], action[1]] == 3:
            boardHits.append(action)

            for ship in ships:
                if (action, True) in ship:
                    ship[ship.index((action, True))] = (action, False)
                    shipSunk = True
                    for cell in ship:
                        if cell[1]:
                            shipSunk = False
                            break

                    if shipSunk:
                        for cell in ship:
                            boardHits.remove(cell[0])

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
                        board[row, col + i] = 1
                        shipsList[-1].append(((row, col + 1), True))
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
                        board[row + i, col] = 1
                        shipsList[-1].append(((row + 1, col), True))
                    placed = True

    return board, shipsList

def calculateFeatures(state, action, player):
    results = np.array([
        1,
        distanceToHit(state, action, player),
        #distanceToMiss(state, action, player)
    ])

    return results

def distanceToHit(state, action, player):
    # We're interested in the opponent's board
    boardHits = state.playerTwoBoardHits if player == 1 else state.playerOneBoardHits

    minDist = state.boardSize * 2
    for hit in boardHits:
        # Manhattan distance
        tempDist = abs(action[0] - hit[0]) + abs(action[1] - hit[1])
        if tempDist < minDist:
            minDist = tempDist

    #print(state.playerTwoBoard)
    #print(action)
    #print(minDist)
    #print()

    return minDist

def distanceToMiss(state, action, player):
    # We're interested in the opponent's board
    boardMisses = state.playerTwoBoardMisses if player == 1 else state.playerOneBoardMisses

    minDist = state.boardSize * 2
    for miss in boardMisses:
        # Euclidian distance
        tempDist = abs(action[0] - miss[0]) ** 2 + (action[1] - miss[1]) ** 2
        if tempDist < minDist:
            minDist = tempDist

    return minDist
