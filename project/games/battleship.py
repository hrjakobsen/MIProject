from interfaces import IGame
from interface import implements
import numpy as np
import copy
import pygame
import pygame.gfxdraw


class Battleship(implements(IGame)):
    def __init__(self, boardSize=10, ships=[2, 3, 3, 4, 5]):
        """
        :param boardSize: size of the game
        :param ships: list of ship lengths
        """
        self.boardSize = boardSize
        self.ships = ships
        self.p1Game = _BattleshipSingleGame(boardSize, ships)
        self.p2Game = _BattleshipSingleGame(boardSize, ships)
        self.numFeatures = None
        self.turn = None
        self.winner = None

    def __deepcopy__(self, _):
        new = Battleship(self.boardSize, self.ships)
        new.p1Game = copy.deepcopy(self.p1Game)
        new.p2Game = copy.deepcopy(self.p2Game)
        new.numFeatures = self.numFeatures
        new.turn = self.turn
        return new

    def getActions(self, player):
        # Actions available on the opponent's board
        return self.p2Game.getActions() if player == 1 else self.p1Game.getActions()

    def getTurn(self):
        if self.turn is None:
            self.turn = np.random.randint(2) + 1

        return self.turn

    def getNumFeatures(self):
        if self.numFeatures is None:
            self.numFeatures = len(self.p1Game.calculateFeatures((0, 0)))
        return self.numFeatures

    def getFeatures(self, player, action):
        return self.p2Game.calculateFeatures(action) if player == 1 else self.p1Game.calculateFeatures(action)

    def makeMove(self, player, action):
        if player == 1:
            hit = self.p2Game.makeMove(action)
        else:
            hit = self.p1Game.makeMove(action)

        if not hit:
            self.turn = 1 if self.turn == 2 else 2

    def gameEnded(self):
        p1Won = self.p2Game.gameEnded()
        p2Won = self.p1Game.gameEnded()
        if p1Won:
            self.winner = 1
        elif p2Won:
            self.winner = 2

        return p1Won or p2Won

    def getReward(self, player):
        if self.winner == 1 and player == 1:
            return self.p2Game.getReward()
        elif self.winner == 2 and player == 2:
            return self.p1Game.getReward()

        return 0

    def getWinner(self):
        return self.winner

    def draw(self, surface):
        surface.fill(BACKGROUNDCOLOR)
        cellSize = int(min(((surface.get_width() - 10) // 2) / self.boardSize, surface.get_height() / self.boardSize))

        self.p1Game.draw(surface, cellSize, 0)
        self.p2Game.draw(surface, cellSize, surface.get_width() - cellSize * self.boardSize + GRIDSIZE)
        pygame.time.delay(50)


WATER = 0
SHIP = 1
WATERHIT = 2
SHIPHIT = 3
SHIPSUNK = 4

GRIDSIZE = 2
BACKGROUNDCOLOR = (150, 150, 150)
WATERCOLOR = (60, 160, 220)
SHIPCOLOR = (130, 130, 130)
WATERHITCOLOR = (210, 210, 210)
SHIPHITCOLOR = (250, 50, 50)
SHIPSUNKCOLOR = (210, 50, 50)

class _BattleshipSingleGame(object):
    def __init__(self, boardSize=10, ships=[2, 3, 3, 4, 5]):
        self.boardSize = boardSize
        self.actions = None
        self.ships = sorted(ships, reverse=True)
        self.board, self.shipStatus = randomBoard(boardSize, self.ships)
        self.hits = []
        self.numHits = 0
        self.misses = []
        self.numFeatures = None
        self.numTurns = 1
        self.removedShipSquares = []
        self.getActions()

    def __deepcopy__(self, _):
        new = _BattleshipSingleGame(self.boardSize, self.ships)
        new.board = self.board.copy()
        new.shipStatus = copy.deepcopy(self.shipStatus)
        new.hits = self.hits.copy()
        new.numHits = self.numHits
        new.misses = self.misses.copy()
        new.numFeatures = self.numFeatures
        new.numTurns = self.numTurns
        new.getActions()
        return new

    def getActions(self):
        if self.actions is None:
            actions = []
            for (x, y), value in np.ndenumerate(self.board):
                if value == WATER or value == SHIP:
                    actions.append((x, y))
            self.actions = actions

        return self.actions

    def gameEnded(self):
        return not np.any(self.board == SHIP)

    def calculateFeatures(self, action):
        board = []
        for y in range(self.boardSize):
            for x in range(self.boardSize):
                if action[0] == y and action[1] == x:
                    board.append(1)
                elif self.board[y, x] == SHIPHIT:
                    board.append(2)
                elif self.board[y, x] == WATERHIT:
                    board.append(-1)
                elif self.board[y, x] == SHIPSUNK:
                    board.append(3)
                else:
                    board.append(0)

        return board

    def calculateFeaturesOld(self, action):
        results = np.array([
            1,
            distanceToSquares(self, action, self.misses),
            distanceToSquares(self, action, self.hits),
            hitsOnALine(self, action)
        ])

        return results

    def getReward(self):
        return (self.numHits * 20) * (self.boardSize / self.numTurns)

    def makeMove(self, action):
        self.board[action[0], action[1]] += 2
        self.actions.remove(action)
        self.numTurns += 1

        if self.board[action[0], action[1]] == WATERHIT:
            self.misses.append(action)
            return False

        if self.board[action[0], action[1]] == SHIPHIT:
            self.numHits += 1
            self.hits.append(action)
            self.numTurns -= 1

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
                            self.board[cell[0][0], cell[0][1]] = SHIPSUNK
            return True

    def draw(self, surface, sizeModifier, offset):
        for row in range(self.boardSize):
            for col in range(self.boardSize):
                self.drawCell(surface, sizeModifier, offset, row, col, self.board[row][col])

    def drawCell(self, surface, sizeModifier, offset, y, x, content):
        coordinates = (int(x * sizeModifier + offset), int(y * sizeModifier), sizeModifier - GRIDSIZE, sizeModifier - GRIDSIZE)

        if content == WATER or content == WATERHIT:
            color = WATERCOLOR
        elif content == SHIP or content == SHIPHIT:
            color = SHIPCOLOR
        else:
            color = SHIPSUNKCOLOR

        pygame.gfxdraw.box(surface, coordinates, color)

        if content == WATERHIT or content == SHIPHIT:
            if content == WATERHIT:
                hitColor = WATERHITCOLOR
            else:
                hitColor = SHIPHITCOLOR

            radius = (sizeModifier - GRIDSIZE) // 4
            center = (sizeModifier - GRIDSIZE) // 2
            pygame.gfxdraw.filled_circle(surface, coordinates[0] + center, coordinates[1] + center, radius, hitColor)
            pygame.gfxdraw.aacircle(surface, coordinates[0] + center, coordinates[1] + center, radius, hitColor)
            pygame.gfxdraw.filled_circle(surface, coordinates[0] + center, coordinates[1] + center, int(radius * 0.5), color)
            pygame.gfxdraw.aacircle(surface, coordinates[0] + center, coordinates[1] + center, int(radius * 0.5), color)


def randomBoard(boardSize, ships):
    board = np.zeros((boardSize, boardSize), dtype=int)
    shipsList = []

    for ship in ships:
        shipsList.append([])
        placed = False
        while not placed:
            horizontal = np.random.rand() < .5
            placeable = True
            if horizontal:
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
                row = np.random.randint(0, boardSize - ship)
                col = np.random.randint(0, boardSize)
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
    maxDist = state.boardSize * 2
    minDist = maxDist
    for square in squares:
        # Manhattan distance
        tempDist = abs(action[0] - square[0]) + abs(action[1] - square[1])
        if tempDist < minDist:
            minDist = tempDist

    return (minDist - 1) / (maxDist - 1)


def hitsOnALine(state, action):
    count = 0
    for hit in state.hits:
        if action[0] == hit[0] or action[1] == hit[1]:
            count += 1

    return (count - 0) / (state.boardSize - 2)
