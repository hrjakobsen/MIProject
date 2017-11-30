from games.battleshipSingle import BattleshipGame
from agents.QFunctionApproximator import QFunctionApproximator
from agents.RandomAgent import RandomAgent
import matplotlib.pyplot as plt

import numpy as np
import time

np.set_printoptions(suppress=True, precision=20)

def train(agent, numGames, epsilon, boardSize, ships):
    rewards = []
    interval = numGames / 100

    for x in range(numGames):
        game = BattleshipGame(boardSize, ships)

        while not game.gameEnded():
            makeMove(agent, game, 1, epsilon)

        agent.finalize(game, game.getReward(1))
        rewards.append(game.getReward(1))
        #rewards.append(len(np.where(game.playerTwoBoard > 1)[0]))

        if x % interval == 0:
            print("\rTrained %s/%s games" % (x, numGames), end="")

    return rewards

def makeMove(agent, game, player, epsilon):
    actions = game.getActions(player)
    action = agent.getMove(game, game.getReward(player), actions)
    if np.random.rand() < epsilon:
        action = actions[np.random.randint(len(actions))]
        agent.s = None
    game.makeMove(player, action)

np.random.seed(0)

numGames = 1000
boardSize = 6
ships = [5]#, 3, 4, 5]

g = BattleshipGame(boardSize, ships)
agent = QFunctionApproximator(1, g.getActions(1), g.getNumFeatures(), batchSize=1000, gamma=0.9, decay=0.99, alpha=0.1)

startTime = time.time()
outcomes = train(agent, numGames, 0.1, boardSize, ships)

print("\nDone! - Trained on {0} games. Took {1}s.".format(str(numGames), str(round(time.time() - startTime, 2))))

print(outcomes)
print(np.mean(outcomes))
plt.plot(outcomes)
plt.show()