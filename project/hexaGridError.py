from games.hexaGrid import HexaGrid
from agents.qFunctionTabular import QFunctionTabular
from agents.qFunctionSGD import QFunctionSGD
from agents.random import Random
from agents.hexaGridGreedy import HexaGridGreedy
import pickle

import numpy as np
import matplotlib.pyplot as plt
import copy


def loadFromFile(fileName):
    with open(fileName, 'rb') as handle:
        return pickle.load(handle)


def learn(p1, p2, numGames, epsilon, width=3, height=3):
    p2Start = False
    outcomes = []
    interval = numGames / 100

    startGame = HexaGrid(width, height)

    for x in range(numGames):
        game = copy.deepcopy(startGame)

        if p2Start:
            makeMove(p2, game, 2, epsilon)

        while not game.gameEnded():
            makeMove(p1, game, 1, epsilon)
            if game.gameEnded():
                break

            makeMove(p2, game, 2, epsilon)

        agent1.finalize(game)
        agent2.finalize(game)

        p2Start = not p2Start
        outcomes.append(1 if game.getReward(1) == 1 else 2)
        if x % interval == 0:
            print("\rPlayed %s/%s games" % (x, numGames), end="")

    return outcomes


def makeMove(agent, game, player, epsilon):
    action = agent.getMove(game)

    if isinstance(agent, QFunctionTabular):
        errors.append((agent.Q.get((game.hash(), action), 0) - realQs[game.hash(), action]) ** 2)

    if isinstance(agent, QFunctionSGD):
        errors.append((agent.Q(game, action) - realQs[game.hash(), action]) ** 2)

    if np.random.rand() < epsilon:
        actions = game.getActions(player)
        action = actions[np.random.randint(len(actions))]
        agent.s = None
    game.makeMove(player, action)


np.set_printoptions(suppress=True, precision=2)

numGames = 100000
width = 5
height = 5

realQs = loadFromFile("realQ_{0}x{1}".format(width, height))
errors = []

g = HexaGrid(width, height)
#agent1 = QFunctionSGD(1, g.getNumFeatures(), batchSize=1000, gamma=1, decay=0.99, alpha=0.001, minWeight=0, maxWeight=0)
agent1 = QFunctionTabular(1, {}, {}, 1)
agent2 = HexaGridGreedy(2)

np.random.seed(2)
learn(agent1, agent2, numGames, 0.1, width, height)


runningMeanNumber = 100
numDataPoints = 1000

errors = np.convolve(errors, np.ones((runningMeanNumber,))/runningMeanNumber, mode='valid')

pltErrors = []
count = 0
for x in range(0, len(errors), (len(errors)//numDataPoints)):
    print("{0} {1}".format(count, errors[x]))
    pltErrors.append(errors[x])
    count += 1


plt.scatter(range(len(pltErrors)), pltErrors)
plt.show()
