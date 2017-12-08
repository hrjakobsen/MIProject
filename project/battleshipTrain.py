from games.battleshipSingle import BattleshipGame
from agents.QFunctionApproximator import QFunctionApproximator
#import matplotlib.pyplot as plt
import copy
import numpy as np


def train(agent, numGames, numRepeatGames, boardSize, ships, epsilon):
    interval = numGames / 100

    for x in range(numGames):
        if x % numRepeatGames == 0:
            startGame = BattleshipGame(boardSize, ships)

        game = copy.deepcopy(startGame)

        while not game.gameEnded():
            makeMove(agent, game, 1, epsilon)

        agent.finalize(game, game.getReward(1), None)

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


np.set_printoptions(suppress=True, precision=8)

numTrain = 5000
numRepeatGames = 100
trainBoardSize = 6
trainShips = [2, 3, 4]

numPlay = 100
playBoardSize = 6
playShips = [2, 3]#, 3, 4, 5]

np.random.seed(0)
g = BattleshipGame(trainBoardSize, trainShips)
agent1 = QFunctionApproximator(1, g.getNumFeatures(), batchSize=numRepeatGames * trainBoardSize ** 2, gamma=0.9, decay=0.9, alpha=0.1, minWeight=0, maxWeight=0)

np.random.seed(0)
outcomes = play(agent1, numPlay, playBoardSize, playShips)
moves = [sum(playShips) * 20 - outcome for outcome in outcomes]
print("Average score: {0}/{1} | Average moves: {2}/{3}".format(np.mean(outcomes), sum(playShips) * 19, np.mean(moves), sum(playShips)))
#plt.plot(outcomes)
#plt.show()

train(agent1, numTrain, numRepeatGames, trainBoardSize, trainShips, 0.1)

np.random.seed(0)
outcomes = play(agent1, numPlay, playBoardSize, playShips)
moves = [sum(playShips) * 20 - outcome for outcome in outcomes]
print("Average score: {0}/{1} | Average moves: {2}/{3}".format(np.mean(outcomes), sum(playShips) * 19, np.mean(moves), sum(playShips)))#plt.plot(outcomes)
#plt.plot(outcomes)
#plt.show()