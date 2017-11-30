from games.HexagonGame import HexagonGame
from agents.TabularQLearner import TabularQLearner
from agents.QFunctionApproximator import QFunctionApproximator
from agents.RandomAgent import RandomAgent
from agents.GreedyHexAgent import GreedyHexAgent

import numpy as np
import matplotlib.pyplot as plt
import time
import copy

realQs = {}
errors = []

def learn(agent1, agent2, numGames, epsilon, width=3, height=3):
    p2Start = False
    outcomes = []
    interval = numGames / 100

    startGame = HexagonGame(width, height)

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
        errors.append(agent.Q.get((game.hash(), action), 0) - realQs[game.hash(), action])

    if np.random.rand() < epsilon:
        action = actions[np.random.randint(len(actions))]
        agent.s = None
    game.makeMove(player, action)


np.set_printoptions(suppress=True, precision=2)
np.random.seed(0)

numGames = 100
width = 5
height = 5

g = HexagonGame(width, height)
agent1 = TabularQLearner({}, {}, 0.9)
agent2 = RandomAgent()

learn(agent1, agent2, numGames, 0.5, width, height)

print(errors)

plt.plot(errors)
plt.show()
