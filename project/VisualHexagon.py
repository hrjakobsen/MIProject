from agents.GreedyHexAgent import GreedyHexAgent
from agents.QFunctionApproximator import QFunctionApproximator
from games.hexagon import HexagonGame
from agents.TabularQLearner import TabularQLearner
from agents.RandomAgent import RandomAgent

import pygame
import numpy as np
import copy
import time

gameSizeModifier = 150

def makeMove(agent, game, player, epsilon):
    action = agent.getMove(game, game.getReward(player), game.getActions())
    if np.random.rand() < epsilon:
        action = g.getActions()[np.random.randint(len(game.getActions()))]
        agent.s = None
    game.makeMove(player, action)


np.set_printoptions(suppress=True, precision=2)
np.random.seed(0)
g = HexagonGame(1, 1)

numGames = 1
width = 3
height = 3

def drawHexagon(game: HexagonGame, surface):
    for x in range(game.width):
        for y in range(game.height + 1):
            cell = game.board[(y, x)]
            if cell == -1: continue
            drawCell(surface, x, y, cell)


def drawCell(surface, x, y, colour):
    colours = [
        (255, 0, 0),
        (0, 255, 0),
        (0, 0, 255),
        (255, 255, 0),
        (255, 140, 0),
        (0, 0, 0),
        (255, 255, 255)
    ]

    foundColour = 0
    if colour < 5:
        foundColour = colours[colour]
    elif colour < 10:
        foundColour = colours[5]
    else:
        foundColour = colours[6]

    pygame.draw.circle(surface,
                       foundColour,
                       (int(x * gameSizeModifier + gameSizeModifier // 2),
                        int(y * gameSizeModifier + gameSizeModifier // 2 - (x % 2) * gameSizeModifier // 2)),
                       int(gameSizeModifier // 3))


def learnVisual(gameWidth, gameHeight, epsilon=0.001):
    pygame.init()

    drawHeight = (gameHeight + 1) * gameSizeModifier
    drawWidth = gameWidth * gameSizeModifier
    pygame.display.set_mode((drawWidth, drawHeight))

    surface = pygame.display.get_surface()

    game = HexagonGame(gameWidth, gameHeight)
    runningGame = copy.deepcopy(game)

    isRunning = True

    agent1 = GreedyHexAgent(1) #QFunctionApproximator(2, len(game.getFeatures(2)), game.getActions(), gamma=1, batchSize=100)#RandomAgent(g.getActions())
    agent2 = TabularQLearner({}, {}, 1)

    playerTurn = 1

    gamesPlayed = 0
    player1Won = 0

    drawGame = False

    while isRunning:
        if drawGame:
            surface.fill((200, 200, 200))
            drawHexagon(runningGame, surface)

        makeMove(agent1 if playerTurn == 1 else agent2, runningGame, playerTurn, epsilon)

        if runningGame.gameEnded():
            reward1 = runningGame.getReward(1)
            reward2 = runningGame.getReward(2)

            agent1.finalize(runningGame, reward1, game.getActions())
            agent2.finalize(runningGame, reward2, game.getActions())
            runningGame = copy.deepcopy(game)
            player1Won += 1 if reward1 == 1 else 0
            gamesPlayed += 1
            pygame.display.set_caption("player 1 won {0}, out of {1} games".format(player1Won, gamesPlayed))

        playerTurn = 2 if playerTurn == 1 else 1

        pygame.display.flip()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                isRunning = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    isRunning = False
                elif event.key == pygame.K_p:
                    drawGame = not drawGame

        if drawGame:
            pygame.time.delay(100)

learnVisual(width, height)
