import pygame
import numpy as np
import time
import math

from agents.RandomAgent import RandomAgent
from agents.GreedyPongAgent import GreedyPongAgent
from agents.QFunctionApproximator import QFunctionApproximator
from games.pong import PongGame, makeMove, pongWidth, pongHeight

np.set_printoptions(suppress=True, precision=4  )
np.random.seed(0)
paddleDrawWidth = 4
gameSizeModifier = 8
frame = 0

def drawGame(surface, game: PongGame, p1wins, numgames, myfont):
    global frame
    frequency = 0.01

    # paddle 1
    pygame.draw.rect(surface, (30, 160, 0),
                     (0,
                      int(game.p1pos - game.paddleHeight // 2) * gameSizeModifier,
                      paddleDrawWidth * gameSizeModifier,
                      game.paddleHeight * gameSizeModifier))

    pygame.draw.rect(surface, (0, 0, 255),
                     (paddleDrawWidth * 2 + pongWidth * gameSizeModifier,
                      int(game.p2pos - game.paddleHeight // 2) * gameSizeModifier,
                      paddleDrawWidth * gameSizeModifier,
                      game.paddleHeight * gameSizeModifier))

    bx, by = game.ballPosition * gameSizeModifier + paddleDrawWidth
    pygame.draw.circle(surface, (255, 50, 50),
                       (int(bx + game.ballRadius * gameSizeModifier), int(by)),
                       int(game.ballRadius * gameSizeModifier))

    text = myfont.render(str(p1wins), 1, (0, 0, 0))
    surface.blit(text, (game.width * gameSizeModifier // 2 + paddleDrawWidth - game.width * gameSizeModifier // 4,
                        pongHeight * gameSizeModifier // 8))

    text = myfont.render(str(numgames - p1wins), 1, (0, 0, 0))
    surface.blit(text, (game.width * gameSizeModifier // 2 + paddleDrawWidth + pongWidth * gameSizeModifier // 4,
                        pongHeight * gameSizeModifier // 8))

    frame += 1

def drawTraining(surface, myfont, numgames):
    text = myfont.render("Training...", 1, (255, 0, 0))
    surface.blit(text, (pongWidth * 0.5, pongHeight * gameSizeModifier // 2))
    text = myfont.render("game {0}".format(str(numgames)), 1, (255, 0, 0))
    surface.blit(text, (pongWidth * 0.5, pongHeight * gameSizeModifier // 6))


def getMove(agent, game: PongGame, player, epsilon):
    actions = game.getActions(player)
    action = agent.getMove(game, game.getReward(player), actions)
    if np.random.rand() < epsilon:
        action = actions[np.random.randint(len(actions))]
        agent.s = None
    return action

def getMove2(agent, game: PongGame, player):
    return agent.getTrainedMove(game, game.getActions(player))

def learnPong(epsilon):
    pygame.init()
    pygame.display.set_mode((pongWidth * gameSizeModifier + paddleDrawWidth * 2 * gameSizeModifier, pongHeight * gameSizeModifier))
    surface = pygame.display.get_surface()

    myfont = pygame.font.SysFont("monospace", 15 * gameSizeModifier)

    p1Wins = 0
    numGames = 0
    isRunning = True
    displayGame = True

    game = PongGame()

    features = game.getNumFeatures()
    #agent1 = RandomAgent()
    agent1 = QFunctionApproximator(1, features, gamma=1, batchSize=4000, alpha=0.1, minWeight=-20, maxWeight=-20)
    agent2 = RandomAgent()#GreedyPongAgent(2)#RandomAgent()#QFunctionApproximator(2, features, gamma=0.9, batchSize=1000, alpha=0.2, minWeight=-20, maxWeight=-20)

    while isRunning:
        if displayGame:
            surface.fill((200, 200, 200))
            drawGame(surface, game, p1Wins, numGames, myfont)
            pygame.display.flip()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                isRunning = False
            elif event.type == pygame.KEYDOWN:
                # check if p
                if event.key == pygame.K_p:
                    displayGame = not displayGame
                    if not displayGame:
                        surface.fill((200, 200, 200))
                        drawTraining(surface, myfont, numGames)
                        pygame.display.flip()
                elif event.key == pygame.K_q:
                    isRunning = False

        if numGames < 200:
            action1 = getMove(agent1, game, 1, epsilon)
            action2 = getMove(agent2, game, 2, epsilon)
        else:
            action1 = getMove2(agent1, game, 1)
            action2 = getMove2(agent2, game, 2)

        game = makeMove(game, action1, action2)

        if game.gameEnded():
            agent1Reward = game.getReward(1)
            agent2Reward = game.getReward(2)
            agent1.finalize(game, agent1Reward, game.getActions(1))
            agent2.finalize(game, agent2Reward, game.getActions(2))
            p1Wins += 1 if agent1Reward > 0 else 0
            numGames += 1

            if(numGames == 200 or numGames == 400):
                print("Player 1 wins: ", p1Wins)
                p1Wins = 0

            game = PongGame()

            if not displayGame:
                surface.fill((200, 200, 200))
                drawTraining(surface, myfont, numGames)
                pygame.display.flip()

        if displayGame:
            pygame.time.delay(20)

    startTime = time.time()
    print("\nDone! - Played {0} games. Took {1}s. Won {2} games.".format(str(numGames), str(
        round(time.time() - startTime, 2)), str(p1Wins)))
    pygame.quit()

learnPong(0.1)
