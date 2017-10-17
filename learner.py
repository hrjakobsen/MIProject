import numpy as np
np.random.seed(0)

def randomGame(width, height):
    board = np.random.randint(5, size=(height + 1, width))
    
    # What we actually want is a jagged array where the odd cols 
    # are shifted half a cell up, to simulate a hexagon grid.
    # We achieve this by making the grid 1 higher than specified and 
    # 'remove' the top cell in the odd columns
    for x in range(width // 2):
        board[0][x * 2 + 1] = -1
        
    # Set the initial player positions
    board[0, 0] = 5
    board[height, width - 1] = 10
    
    return board

def getHash(board):
    # This function generates a unique hash for each board
    # This is done by realising that each cell on the board
    # can be in one of 15 states (5 colors * 3 owned-states)
    # This means that each cell can be stored in a single 
    # base-16 character
       
    return ''.join([format(int(i), 'x') if i != - 1 else "" for i in np.nditer(board.T)])

def makeMove(board, player, action):
    # Do everything on a copy to ensure stateless-ness
    board = board.copy()
    
    height = board.shape[0]
    width = board.shape[1]
    
    frontier = getOwnedCells(board, player)
    
    # Our color can spread through cells that were just added
    # so we maintain a frontier that is the cells that we still
    # need to check for neighbors of the right colour
    while len(frontier):
        point = frontier.pop()
        board[point[0], point[1]] = action + player * 5
        
        neighbours = pointsAround(point)
        
        # Find the neighbours that are inside the board and
        # have the color of the action and add them to the frontier 
        for neighbour in neighbours:
            y, x = neighbour[0], neighbour[1]

            if (0 <= y < height 
                and 0 <= x < width 
                and board[y, x] != -1 
                and board[y, x] == action):
                frontier.append(neighbour)
    
    # If a player has no more moves, the other player is rewarded
    # the rest of the cells on the board
    board = finaliseBoard(board, player)
    return board

def getOwnedCells(board, player):
    lowerLimit = player * 5
    upperLimit = (player + 1) * 5
    
    frontier = list(
        np.column_stack(
            np.where(
                np.logical_and(board < upperLimit, board >= lowerLimit)
            )
        )
    )
    
    return frontier

def finaliseBoard(board, playerCall):
    board = board.copy()
    player = 2 if playerCall == 1 else 1
    frontier = getOwnedCells(board, player)
    height = board.shape[0]
    width = board.shape[1]
    
    playerColor = board[0, 0] if playerCall == 1 else board[height - 1, width - 1]
    found = False
    
    # Check if the player has a valid action that gains more cells
    for cell in frontier:
        neighbours = pointsAround(cell)
        for neighbour in neighbours:
            y, x = neighbour[0], neighbour[1]

            if (0 <= y < height 
                and 0 <= x < width 
                and 0 <= board[y, x] < 5):
                found = True
                break
        if found:
            break
    if found:
        return board
    
    # Award the player the rest of the cells
    for x in np.nditer(board, op_flags=['readwrite']):
        if 0 <= int(x) < 5:
            x[...] = playerColor
    return board

def getReward(game, player):
    # This function calculates the reward of a game
    # We reward nothing for a move unless it is a winning move
    # Then it gains 1 point. If it is a losing move, it gains -1
    if not gameEnded(game):
        return 0

    height = game.shape[0]
    width = game.shape[1]
    
    player1Color = game[0,0]
    player2Color = game[height - 1, width - 1]
    player1count = np.count_nonzero(game == player1Color)
    player2count = np.count_nonzero(game == player2Color)
    
    if player == 1:
        if player1count > player2count:
            return 1
        else:
            return -1
    else:
        if player2count > player1count:
            return 1
        else:
            return -1

def gameEnded(board):
    # Simply checks if there are any cells that are not owned 
    return not np.any(np.logical_and(board >= 0, board < 5))

def pointsAround(point):
    y, x = point[0], point[1]
    odd = x % 2 == 1
    neighbours = []
    
    relOddCoords = [
        (-1, 0),
        (-1, 1),
        (0, 1),
        (1, 0),
        (0, -1),
        (-1, -1)
    ]

    relEvenCoords = [
        (-1, 0),
        (0, 1),
        (1, 1),
        (1, 0),
        (1, -1),
        (0, -1)
    ]
    
    for coord in relOddCoords if odd else relEvenCoords:
        newY = y + coord[0]
        newX = x + coord[1]
        neighbours.append([newY, newX])
    return neighbours


class Board(object):

    def __init__(self, height, width):
        self.height = height
        self.width = width

class HexLearner(object):
    # This class is an implementation of the Q-Learning-Agent
    # from Russel & Norvig (2010) p. 844
    
    def __init__(self, player, Q = {}, N = {}):
        self.Q = Q
        self.N = N
        self.s = None
        self.a = None
        self.r = None
        self.player = player
        self.actions = [0, 1, 2, 3, 4]
    
    def getMove(self, state, reward):
        if gameEnded(state):
            self.Q[getHash(state), None] = reward
        if self.s is not None:
            self.incrementN()
            self.updateQ(state, reward)
        self.s = state
        self.a = self.argmax(state)
        self.r = reward
        return self.a
    
    def initializeQ(self, s, a):
        if (s, a) not in self.Q:
            self.Q[s, a] = 0
    
    def initializeN(self, s, a):
        if (s, a) not in self.N:
            self.N[s, a] = 0
    
    def argmax(self, state):
        vals = []
        s = getHash(state)
        for a in self.actions:
            self.initializeQ(s, a)
            self.initializeN(s, a)
            vals.append(self.f(self.Q[s, a], self.N[s, a]))
        return self.actions.index(vals.index(max(vals)))
            
    
    def f(self, val, num):
        if num < 1:
            return 1
        return val
    
    def incrementN(self):
        s = getHash(self.s)
        a = self.a
        self.initializeN(s, a)
        self.N[s, a] += 1
    
    def updateQ(self, sP, rP):
        s = getHash(self.s)
        a = self.a
        self.initializeQ(s, a)
        self.Q[s, a] = self.Q[s, a] + self.alpha() * (self.r + self.getBestQToSp(sP) - self.Q[s, a])
        
    def alpha(self):
        return 1 / self.N[getHash(self.s), self.a]
    
    def getBestQToSp(self, sP):
        newState = getHash(sP)
        bestAction = None
        bestQ = -1000000000000
        for a in self.actions:
            self.initializeQ(newState, a)
            if self.Q[newState, a] > bestQ:
                bestAction = a
                bestQ = self.Q[newState, a]
        return bestQ

import time
from IPython.display import clear_output

def learn(trials, epsilon):
    NUMTRIALS = trials
    EPSILON = epsilon
    latestQ1 = {}
    latestN1 = {}
    latestQ2 = {}
    latestN2 = {}

    startTime = time.time()
    startGame = randomGame(3, 3)

    for x in range(NUMTRIALS):
        agent1 = HexLearner(1, latestQ1, latestN1)
        agent2 = HexLearner(2, latestQ2, latestN2)
        #game = randomGame(3,3)
        game = startGame.copy()
        while not gameEnded(game):
            num = np.random.rand()
            action = agent1.getMove(game, getReward(game, 1))
            if num < EPSILON:
                action = np.random.randint(5)
            game = makeMove(game, 1, action)
            
            action = agent2.getMove(game, getReward(game, 2))
            if num < EPSILON:
                action = np.random.randint(5)
            game = makeMove(game, 2, action)
            
        agent1.getMove(game, getReward(game, 1))
        agent1.getMove(game, getReward(game, 1))
        agent2.getMove(game, getReward(game, 2))
        latestQ1 = agent1.Q
        latestN1 = agent1.N
        latestQ2 = agent2.Q
        latestN2 = agent2.N

        #if x % (NUMTRIALS / 1000) == 0 and x > 0:
            #elapsedSec = time.time() - startTime
            #gamesPerSec = x / elapsedSec
            #remainingSec = str(round((NUMTRIALS - x) / gamesPerSec, 2))
            #percentDone = str(round(x / NUMTRIALS * 100, 2))

            #clear_output(wait=True)
            #print(percentDone + "% done - " + remainingSec + "s remaining")

    print("\nDone! - Took " + str(round(time.time() - startTime, 2)) + "s. Saw " + str(len(latestQ1)) + " states")
    return latestQ1, latestN1  

import cProfile, pstats
np.random.seed(0)

pr = cProfile.Profile()
pr.enable()
latestQ1, latestN1 = learn(5000, 0.1)
pr.disable()
ps = pstats.Stats(pr).sort_stats('tottime')
ps.print_stats()

#print(latestQ1)