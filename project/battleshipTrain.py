from games.battleshipSingle import BattleshipGame
from agents.QFunctionApproximator import QFunctionApproximator
#import matplotlib.pyplot as plt

import numpy as np

np.set_printoptions(suppress=True, precision=8)


def train(agent, numGames, epsilon, boardSize, ships):
    interval = numGames / 100

    for x in range(numGames):
        game = BattleshipGame(boardSize, ships)

        while not game.gameEnded():
            makeMove(agent, game, 1, epsilon)

        agent.finalize(game, game.getReward(1))

        if x % interval == 0:
            print("\rTrained %s/%s games %s" % (x, numGames, agent.weights), end="")
    print()


def play(agent, numGames, boardSize, ships):
    rewards = []
    interval = numGames / 100

    for x in range(numGames):
        game = BattleshipGame(boardSize, ships)

        while not game.gameEnded():
            game.makeMove(1, agent.getTrainedMove(game, game.getActions(1)))

        rewards.append(game.getReward(1))

        if x % interval == 0:
            print("\rPlayed %s/%s games" % (x, numGames), end="")
    print()

    return rewards

def makeMove(agent, game, player, epsilon):
    actions = game.getActions(player)
    action = agent.getMove(game, game.getReward(player), actions)
    if np.random.rand() < epsilon:
        action = actions[np.random.randint(len(actions))]
        agent.s = None
    game.makeMove(player, action)


numTrain = 1000
trainBoardSize = 6
trainShips = [2, 3, 4]

numPlay = 100
playBoardSize = trainBoardSize
playShips = trainShips

g = BattleshipGame(trainBoardSize, trainShips)
agent = QFunctionApproximator(1, g.getNumFeatures(), batchSize=1000, gamma=0.9, decay=0.95, alpha=0.1)

np.random.seed(0)
outcomes = play(agent, numPlay, playBoardSize, playShips)
moves = [sum(playShips) * 20 - outcome for outcome in outcomes]
print("Average score: {0}/{1} ({2}) | Average moves: {3}/{4} ({5}) | Deviation: {6}".format(np.mean(outcomes), sum(playShips) * 19, round(np.mean(outcomes) / (sum(playShips) * 19), 2), np.mean(moves), sum(playShips), round(np.mean(moves) / sum(playShips), 2), np.std(outcomes)))
#plt.plot(outcomes)
#plt.show()

train(agent, numTrain, 0.1, trainBoardSize, trainShips)

np.random.seed(0)
outcomes = play(agent, numPlay, playBoardSize, playShips)
moves = [sum(playShips) * 20 - outcome for outcome in outcomes]
print("Average score: {0}/{1} ({2}) | Average moves: {3}/{4} ({5}) | Deviation: {6}".format(np.mean(outcomes), sum(playShips) * 19, round(np.mean(outcomes) / (sum(playShips) * 19), 2), np.mean(moves), sum(playShips), round(np.mean(moves) / sum(playShips), 2), np.std(outcomes)))
#plt.plot(outcomes)
#plt.show()