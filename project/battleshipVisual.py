from games.battleshipSingle import BattleshipGame
from agents.QFunctionApproximator import QFunctionApproximator
from agents.HuntAndTargetAgent import HuntAndTargetAgent
from agents.RandomAgent import RandomAgent

import pygame
import numpy as np
import copy

gameSizeModifier = 30

def makeMove(agent, game, epsilon):
    actions = game.getActions(1)
    action = agent.getMove(game, game.getReward(1), actions)
    if np.random.rand() < epsilon:
        action = actions[np.random.randint(len(actions))]
        agent.s = None
    game.makeMove(1, action)

def makeTrainedMove(agent, game):
    actions = game.getActions(1)
    game.makeMove(1, agent.getTrainedMove(game, actions))

def drawBattleship(game: BattleshipGame, surface):
    for row in range(game.boardSize):
        for col in range(game.boardSize):
            drawCell(surface, row, col, game.board[row][col])


def drawCell(surface, y, x, content):
    coordinates = (int(x * gameSizeModifier + gameSizeModifier // 2) - ((gameSizeModifier // 2)) , int(y * gameSizeModifier + gameSizeModifier // 2) - (gameSizeModifier // 2), gameSizeModifier, gameSizeModifier)

    shot = content == 2 or content == 3

    if content == 0 or content == 2:
        colour = (0, 0, 255)

    if content == 1 or content == 3:
        colour = (100, 100, 100)

    pygame.draw.rect(surface, colour, coordinates)

    if shot:
        pygame.draw.circle(surface, (255, 0, 0), (int(x * gameSizeModifier + gameSizeModifier // 2), int(y * gameSizeModifier + gameSizeModifier // 2)), (gameSizeModifier // 10))


def trainAgent(agent, numGames, boardSize, ships, epsilon):
    i = 0
    StartGame = BattleshipGame(boardSize, ships)
    for x in range(numGames):
        #print(agent.weights)
        if i < numGames/10:
            game = copy.deepcopy(StartGame)
        else:
            game = BattleshipGame(boardSize, ships)
            #print(agent.weights)
            i = 0

        while not game.gameEnded():
            makeMove(agent, game, epsilon)
        agent.finalize(game, game.getReward(1), game.getActions(1))
        i += 1
        print("Remaining games: ", numGames-x, agent.weights)


def learnVisual(agents, numGames, boardSize, ships, epsilon):
    pygame.init()

    drawHeight = boardSize * gameSizeModifier
    drawWidth = boardSize * gameSizeModifier
    pygame.display.set_mode((drawWidth, drawHeight))

    surface = pygame.display.get_surface()
    drawGame = True
    funcApproxWins = 0
    numOfDraws = 0

    #startGame = BattleshipGame(boardSize, ships)

    playerMoves = [0, 0, 0, 0]
    for x in range(numGames):
        startGame = BattleshipGame(boardSize, ships)
        playerMoves[0] = 0
        playerMoves[1] = 0
        playerIndex = 0
        for agent in agents:
            numMoves = 0
            pygame.display.set_caption("Game {0} - {1}".format(x, agent.weights if isinstance(agent, QFunctionApproximator) else []))
            #if x % 100 == 99:
            #    startGame = BattleshipGame(boardSize, ships)

            game = copy.deepcopy(startGame)
            #game = BattleshipGame(boardSize, ships)

            while not game.gameEnded():
                makeTrainedMove(agent, game)

                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        return
                    elif event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_q:
                            return
                        elif event.key == pygame.K_p:
                            drawGame = not drawGame

                if drawGame:
                    pygame.time.delay(1000)

                if drawGame:
                    surface.fill((200, 200, 200))
                    drawBattleship(game, surface)
                    pygame.display.flip()

            playerMoves[playerIndex] = game.numMoves
            playerMoves[playerIndex+2] += game.numMoves
            playerIndex += 1
            #agent.finalize(game, game.getReward(1), game.getActions(1))
        if playerMoves[0] < playerMoves[1]:
            funcApproxWins += 1
        if playerMoves[0] == playerMoves[1]:
            numOfDraws += 1

    print("func wins: ", funcApproxWins)
    print("num of draws: ", numOfDraws)
    print("avg. num of moves for func: ", playerMoves[2] / numGames)
    print("avg. num of moves for HuntAndTarget: ", playerMoves[3] / numGames)



numTrain = 1000
trainBoardSize = 10
trainShips = [2, 3, 3, 4, 5]

g = BattleshipGame(trainBoardSize, trainShips)
#agent = QFunctionApproximator(1, g.getNumFeatures(), batchSize=1000, gamma=0.9, decay=0.98, alpha=0.1, minWeight=-1, maxWeight=1)
agent1 = QFunctionApproximator(1, g.getNumFeatures(), batchSize=80, gamma=0.9, decay=0.98, alpha=0.5, minWeight=0, maxWeight=0)
agent2 = HuntAndTargetAgent(trainBoardSize)
#agent2 = RandomAgent()

agents = [agent1, agent2]

np.random.seed(1)

#trainAgent(agent1, numTrain, trainBoardSize, trainShips, 0.1)

agents[0].weights[1] = 4
agents[0].weights[2] = -8
agents[0].weights[3] = 4
learnVisual(agents, numTrain, trainBoardSize, trainShips, 0)

