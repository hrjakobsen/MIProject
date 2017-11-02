from games.HexagonGame import HexagonGame
from agents.HexagonQLearner import HexLearner
from agents.HexagonRandom import HexRandom
from agents.HexagonFALearner import HexFALearner
import numpy as np
import time
import copy
import os.path

def learn(agent1, agent2, numGames, epsilon, width, height):
    p2Start = False
    p1Wins = 0
    interval = numGames / 100

    startGame = HexagonGame(width, height)

    for x in range(numGames):
        #game = HexagonGame(width, height)
        game = copy.deepcopy(startGame)

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
    game.makeMove(player, action)

np.random.seed(0)

width = 5
height = 5

agent1 = HexFALearner(1, width, height, 50)
print(agent1.weights)
print(len(agent1.weights))
agent2 = HexRandom()

numGames = 100
startTime = time.time()

wins = learn(agent1, agent2, numGames, 0.1, width, height)

print("\nDone! - Played {0} games. Took {1}s. Won {2} games.".format(str(numGames), str(
    round(time.time() - startTime, 2)), str(wins)))

print(agent1.weights)