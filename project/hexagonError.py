from games.hexagon import HexagonGame, getHash
from agents.TabularQLearner import TabularQLearner
from agents.QFunctionApproximator import QFunctionApproximator
from agents.RandomAgent import RandomAgent
from agents.GreedyHexAgent import GreedyHexAgent
import pickle

import numpy as np
import matplotlib.pyplot as plt
import copy


def loadFromFile(fileName):
    with open(fileName, 'rb') as handle:
        return pickle.load(handle)


def learn(agent1, agent2, numGames, epsilon, width=3, height=3):
    p2Start = False
    outcomes = []
    interval = numGames / 100

    startGame = HexagonGame(width, height)

    for x in range(numGames):
        game = copy.deepcopy(startGame)

        if p2Start:
            makeMove(agent2, game, 2, epsilon)

        while not game.gameEnded():
            makeMove(agent1, game, 1, epsilon)
            if game.gameEnded():
                break

            makeMove(agent2, game, 2, epsilon)

        agent1.finalize(game, game.getReward(1), game.getActions())
        agent2.finalize(game, game.getReward(2), game.getActions())

        p2Start = not p2Start
        outcomes.append(1 if game.getReward(1) == 1 else 2)
        if x % interval == 0:
            print("\rPlayed %s/%s games" % (x, numGames), end="")

    return outcomes


def makeMove(agent, game, player, epsilon):
    actions = game.getActions()
    action = agent.getMove(game, game.getReward(player), actions)

    if agent == agent1:
        errors.append((agent.Q.get((game.hash(), action), 0) - realQs[game.hash(), action]) ** 2)
        #errors.append((agent.Q(game, action) - realQs[game.hash(), action]) ** 2)

    if np.random.rand() < epsilon:
        action = actions[np.random.randint(len(actions))]
        agent.s = None
    game.makeMove(player, action)


np.set_printoptions(suppress=True, precision=2)

numGames = 10000
width = 5
height = 5

realQs = loadFromFile("realQ_{0}x{1}".format(width, height))
errors = []

g = HexagonGame(width, height)
#agent1 = QFunctionApproximator(1, len(g.calculateFeatures(g, 0, 1)), batchSize=1000, gamma=1, decay=0.99, alpha=0.1)
agent1 = TabularQLearner({}, {}, 1)
agent2 = GreedyHexAgent(2)

np.random.seed(2)
learn(agent1, agent2, numGames, 0.1, width, height)



runningMeanNumber = 100
numDataPoints = 100

errors = np.convolve(errors, np.ones((runningMeanNumber,))/runningMeanNumber, mode='valid')
count = 0
for x in range(0, len(errors), (len(errors)//numDataPoints)):
    print("{0} {1}".format(count, errors[x]))
    count += 1

plt.scatter(range(len(errors)), errors)
plt.show()
