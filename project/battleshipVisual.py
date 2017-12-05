from games.battleshipSingle import BattleshipGame
from agents.QFunctionApproximator import QFunctionApproximator

import pygame
import numpy as np
import copy

gameSizeModifier = 150

def makeMove(agent, game, epsilon):
    actions = game.getActions(1)
    action = agent.getMove(game, game.getReward(1), actions)
    if np.random.rand() < epsilon:
        action = actions[np.random.randint(len(actions))]
        agent.s = None
    game.makeMove(1, action)


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


def learnVisual(agent, numGames, boardSize, ships, epsilon):
    pygame.init()

    drawHeight = boardSize * gameSizeModifier
    drawWidth = boardSize * gameSizeModifier
    pygame.display.set_mode((drawWidth, drawHeight))

    surface = pygame.display.get_surface()
    drawGame = True

    for x in range(numGames):
        game = BattleshipGame(boardSize, ships)

        if drawGame:
            surface.fill((200, 200, 200))
            drawBattleship(game, surface)
            pygame.display.flip()

        while not game.gameEnded():
            makeMove(agent, game, epsilon)

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_q:
                        return
                    elif event.key == pygame.K_p:
                        drawGame = not drawGame

            if drawGame:
                pygame.time.delay(100)

            if drawGame:
                surface.fill((200, 200, 200))
                drawBattleship(game, surface)
                pygame.display.flip()

        agent.finalize(game, game.getReward(1), game.getActions(1))

numTrain = 1000
trainBoardSize = 6
trainShips = [2, 3]

g = BattleshipGame(trainBoardSize, trainShips)
agent = QFunctionApproximator(1, g.getNumFeatures(), batchSize=1000, gamma=0.9, decay=0.95, alpha=0.1, weightMultiplier=1)

np.random.seed(0)

learnVisual(agent, numTrain, trainBoardSize, trainShips, 0)

