from games.hexagon import HexagonGame, getHash
from agents.TabularQLearner import TabularQLearner
from agents.QFunctionApproximator import QFunctionApproximator
from agents.RandomAgent import RandomAgent
from agents.GreedyHexAgent import GreedyHexAgent
import pickle

import numpy as np
import matplotlib.pyplot as plt
import time
import copy

def loadFromFile(fileName):
    with open(fileName, 'rb') as handle:
        return pickle.load(handle)

realQs = loadFromFile("Bruteforce.txt")
errors = []

def learn(agent1, agent2, numGames, epsilon, width=3, height=3):
    p2Start = False
    outcomes = []
    interval = numGames / 100

    startGame = HexagonGame(width, height)
    startGame.board = np.array([ np.array([5, -1, 3, -1, 3]), np.array([2, 4, 0, 0, 4]), np.array([2, 1, 0, 1, 1]), np.array([0, 1, 4, 3, 0]), np.array([3, 0, 2, 3, 0]), np.array([1, 3, 3, 3, 10])])
    startGame._hash = getHash(startGame.board)

    print(startGame.board)

    for x in range(numGames):
        game = copy.deepcopy(startGame)
        # game = HexagonGame(width, height)

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

    if np.random.rand() < epsilon:
        action = actions[np.random.randint(len(actions))]
        agent.s = None
    game.makeMove(player, action)


np.set_printoptions(suppress=True, precision=2)
np.random.seed(0)

numGames = 1000
width = 5
height = 5

g = HexagonGame(width, height)
agent1 = TabularQLearner(realQs, {}, 1)
agent2 = GreedyHexAgent(2)

learn(agent1, agent2, numGames, 0.1, width, height)

print()
print(len(errors))
print(errors)

plt.scatter(range(len(errors)), errors)
plt.show()
