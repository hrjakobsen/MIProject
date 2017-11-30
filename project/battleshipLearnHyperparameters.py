from games.battleship import BattleshipGame
from agents.QFunctionApproximator import QFunctionApproximator

import numpy as np
import time

np.set_printoptions(suppress=True, precision=6)

def train(agent, numGames, epsilon, boardSize, ships):
    rewards = []

    for x in range(numGames):
        game = BattleshipGame(boardSize, ships)

        while not game.gameEnded():
            makeMove(agent, game, 1, epsilon)

        agent.finalize(game, game.getReward(1))

    return rewards

def play(agent, numGames, boardSize, ships):
    rewards = []

    for x in range(numGames):
        game = BattleshipGame(boardSize, ships)

        while not game.gameEnded():
            game.makeMove(1, agent.getTrainedMove(game, game.getActions(1)))

        agent.finalize(game, game.getReward(1))
        rewards.append(game.getReward(1))

    return rewards

def makeMove(agent, game, player, epsilon):
    actions = game.getActions(player)
    action = agent.getMove(game, game.getReward(player), actions)
    if np.random.rand() < epsilon:
        action = actions[np.random.randint(len(actions))]
        agent.s = None
    game.makeMove(player, action)

numTrainGames = 2500
numTestGames = 100
boardSize = 5
ships = [2, 3]

alphas = [0.3, 0.5]
decays = [0.9, 0.5]
gammas = [0.9, 0.8]

results = ["\nAlpha;Decay;Gamma;Weights;Score"]
totalRuns = len(alphas) * len(decays) * len(gammas)
run = 0

for alpha in alphas:
    for decay in decays:
        for gamma in gammas:
            run += 1
            startTime = time.time()
            np.random.seed(0)

            g = BattleshipGame(boardSize, ships)
            agent = QFunctionApproximator(1, g.getActions(1), g.getNumFeatures(), batchSize=1000, gamma=gamma, decay=decay, alpha=alpha)

            train(agent, numTrainGames, 0.1, boardSize, ships)

            outcomes = play(agent, numTestGames, boardSize, ships)

            print("Run {0} of {1} completed. Took {2}s. ETA: {3}".format(str(run), str(totalRuns), round(time.time() - startTime, 2), round((totalRuns - run) *( time.time() - startTime), 2)))
            results.append("{0};{1};{2};{3};{4}".format(alpha, decay, gamma, agent.weights, np.mean(outcomes)))

for res in results:
    print(res)
