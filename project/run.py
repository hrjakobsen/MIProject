from games.HexagonGame import HexagonGame
from agents.HexagonQLearner import HexLearner
from agents.HexagonFunctionApproximator import QFunctionApproximator
from agents.HexagonRandom import HexRandom
from agents.GreedyHexAgent import GreedyHexAgent

import numpy as np
import time
import copy

def learn(agent1, agent2, numGames, epsilon, width=3, height=3):
    p2Start = False
    p1Wins = 0
    interval = numGames / 100

    startGame = HexagonGame(width, height)

    for x in range(numGames):
        game = HexagonGame(width, height)


        if p2Start:
            makeMove(agent2, game, 2, epsilon)

        while not game.gameEnded():
            makeMove(agent1, game, 1, epsilon)
            if game.gameEnded():
                break

            makeMove(agent2, game, 2, epsilon)

        agent1.finalize(game, game.getReward(1))
        agent2.finalize(game, game.getReward(2))
        
        p2Start = not p2Start
        p1Wins += 1 if game.getReward(1) == 1 else 0
        if (x % interval == 0):
            print("\rPlayed %s/%s games" % (x, numGames), end="")

    return p1Wins


def makeMove(agent, game, player, epsilon):
    action = agent.getMove(game, game.getReward(player))
    if np.random.rand() < epsilon:
        action = np.random.randint(5)
        agent.s = None
    game.makeMove(player, action)


np.set_printoptions(suppress=True, precision=2)

np.random.seed(0)
g = HexagonGame(7, 7)
features = len(g.getFeatures(1))
agent1 = QFunctionApproximator(1, features, g.getActions(), gamma=1, batchSize=50)
agent2 = GreedyHexAgent(2)
numGames = 10000

startTime = time.time()
wins = learn(agent1, agent2, numGames, 0.1)

print("\nDone! - Played {0} games. Took {1}s. Won {2} games.".format(str(numGames), str(
    round(time.time() - startTime, 2)), str(wins)))


numGames = 100
startTime = time.time()
wins = learn(agent1, agent2, numGames, 0.1, 11, 11)

print("\nDone! - Played {0} games. Took {1}s. Won {2} games.".format(str(numGames), str(
    round(time.time() - startTime, 2)), str(wins)))


