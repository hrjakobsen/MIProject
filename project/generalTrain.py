from games.battleship import Battleship
from games.hexaGrid import HexaGrid
from games.pong import Pong

from agents.qFunctionSGD import QFunctionSGD
from agents.hexaGridGreedy import HexaGridGreedy
from agents.battleshipHuntAndTarget import BattleShipHuntAndTarget
from agents.pongGreedy import PongGreedy
from agents.random import Random

import pygame
import copy
import numpy as np

BATTLESHIP = 0
HEXAGRID = 1
PONG = 2


def train(p1, p2, numGames, numRepeatGames, gameFunction, epsilon):
    interval = numGames / 100

    for x in range(numGames):
        if x % numRepeatGames == 0:
            startGame = gameFunction()
        game = copy.deepcopy(startGame)

        while not game.gameEnded():
            if game.getTurn() == 1:
                makeMove(p1, game, 1, epsilon)
            else:
                makeMove(p2, game, 2, epsilon)

        p1.finalize(game)
        p2.finalize(game)

        if x % interval == 0:
                print("\rTrained %s/%s games - %s" % (x, numGames, p1.getInfo()), end="")
    print()


def play(p1, p2, numGames, gameFunction, seed=0, visualise=False):
    np.random.seed(seed)
    winners = []
    interval = numGames / 100

    pygame.init()
    pygame.display.set_mode((1000, 750))
    surface = pygame.display.get_surface()

    for x in range(numGames):
        game = gameFunction()
        pygame.display.set_caption("Player 1: {} | Player 2: {} | Draws: {}".format(len([g for g in winners if g == 1]),
                                                                                    len([g for g in winners if g == 2]),
                                                                                    len([g for g in winners if
                                                                                         g == -1])))

        while not game.gameEnded():
            if visualise:
                game.draw(surface)
                pygame.display.flip()

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_t:
                        visualise = not visualise
                    elif event.key == pygame.K_q:
                        pygame.quit()
                        return

            if game.getTurn() == 1:
                game.makeMove(1, p1.getTrainedMove(game))
            else:
                game.makeMove(2, p2.getTrainedMove(game))

        winners.append(game.getWinner())

        if x % interval == 0:
            print("\rPlayed %s/%s games" % (x, numGames), end="")
    print()

    pygame.quit()
    return winners


def makeMove(agent, game, player, epsilon):
    action = agent.getMove(game)
    if np.random.rand() < epsilon:
        actions = game.getActions(player)
        action = actions[np.random.randint(len(actions))]
        agent.s = None
    game.makeMove(player, action)


np.set_printoptions(suppress=True, precision=8)

game = BATTLESHIP
numTrain = 1000
numPlay = 100
numRepeatGames = 5

if game == BATTLESHIP:
    boardSize = 10
    ships = [2, 3, 3, 4, 5]
    gameFunc = lambda: Battleship(boardSize, ships)
    agent2 = Random(2)

if game == HEXAGRID:
    width = 7
    height = 4
    gameFunc = lambda: HexaGrid(width, height)
    agent2 = Random(2)
    agent2 = HexaGridGreedy(2)

if game == PONG:
    width = 100
    height = 50
    gameFunc = lambda: Pong(width, height)
    agent2 = Random(2)

g = gameFunc()
agent1 = QFunctionSGD(1, g.getNumFeatures(), batchSize=100, gamma=1, decay=0.95, alpha=0.001, minWeight=0, maxWeight=0)
#agent2 = QFunctionSGD(2, g.getNumFeatures(), batchSize=100, gamma=1, decay=0.95, alpha=0.001, minWeight=0, maxWeight=0)

outcomes = play(agent1, agent2, numPlay, gameFunc, seed=0, visualise=True)
p1Wins = (len([g for g in outcomes if g == 1]))
print("Player 1 won {} games".format(p1Wins))
print(outcomes)

train(agent1, agent2, numTrain, numRepeatGames, gameFunc, epsilon=.1)

outcomes = play(agent1, agent2, numPlay, gameFunc, seed=0, visualise=True)
p1Wins = (len([g for g in outcomes if g == 1]))
print("Player 1 won {} games".format(p1Wins))
print(outcomes)
