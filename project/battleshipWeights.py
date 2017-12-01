from games.battleshipSingle import BattleshipGame
from agents.QFunctionApproximator import QFunctionApproximator

import numpy as np


def play(agent, numGames, boardSize, ships):
    rewards = []

    for x in range(numGames):
        game = BattleshipGame(boardSize, ships)

        while not game.gameEnded():
            game.makeMove(1, agent.getTrainedMove(game, game.getActions(1)))

        rewards.append(game.getReward(1))

    return rewards


numGames = 10
size = 6
ships = [2, 3, 4]

biasWeights = [-100, -1, 0, 1, 100]
hitDistWeights = [-100, -1, 0, 1, 100]
missDistWeights = [-100, -1, 0, 1, 100]
lineWeights = [-100, -1, 0, 1, 100]

g = BattleshipGame(size, ships)
agent = QFunctionApproximator(1, g.getNumFeatures(), batchSize=1000, gamma=0.9, decay=0.95, alpha=0.1)

print("bias;hitDist;missDist;lineWeight;avg moves")
for biasWeight in biasWeights:
    for hitDistWeight in hitDistWeights:
        for missDistWeight in missDistWeights:
            for lineWeight in lineWeights:
                weights = [biasWeight, hitDistWeight, missDistWeight, lineWeight]
                agent.weights = weights
                np.random.seed(0)
                outcomes = play(agent, numGames, size, ships)
                moves = [sum(ships) * 20 - outcome for outcome in outcomes]
                print("{0};{1};{2};{3};{4}".format(biasWeight, hitDistWeight, missDistWeight, lineWeight, np.mean(moves)))
                #print(weights)
                #print("Average score: {0}/{1} ({2}) | Average moves: {3}/{4} ({5}) | Deviation: {6}".format(np.mean(outcomes), sum(ships) * 19, round(np.mean(outcomes) / (sum(ships) * 19), 2), np.mean(moves), sum(ships), round(np.mean(moves) / sum(ships), 2), np.std(outcomes)))