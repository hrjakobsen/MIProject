from games.battleshipSingle import BattleshipGame
from games.hexagon import HexagonGame
from games.pong import PongGame
from agents.QFunctionApproximator import QFunctionApproximator
import copy
import numpy as np


def train(p1, p2, numGames, numRepeatGames, epsilon, gameFunction):
    interval = numGames / 100
    p1Turn = True

    for x in range(numGames):
        if x % numRepeatGames == 0:
            startGame = gameFunction()
        game = copy.deepcopy(startGame)

        while not game.gameEnded():
            if p1Turn:
                makeMove(p2, game, 2, epsilon)
            else:
                makeMove(p1, game, 1, epsilon)

            p1Turn = not p1Turn

        p1.finalize(game, game.getReward(1), game.getActions())
        p2.finalize(game, game.getReward(2), game.getActions())

        if x % interval == 0:
            print("\rTrained %s/%s games" % (x, numGames), end="")
    print()


def play(p1, p2, numGames, gameFunction):
    rewards = []
    interval = numGames / 100

    p2Start = False

    for x in range(numGames):
        game = gameFunction()

        if p2Start:
            makeMove(p2, game, 2, 0)

        while not game.gameEnded():
            makeMove(p1, game, 1, epsilon)
            if game.gameEnded():
                break

            makeMove(p2, game, 2, epsilon)

        p1.finalize(game, game.getReward(1), game.getActions())
        p2.finalize(game, game.getReward(2), game.getActions())

        p2Start = not p2Start

    for x in range(numGames):
        game = gameFunction()

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


def makeMove(agent, game, player, epsilon):
    actions = game.getActions(player)
    action = agent.getMove(game, game.getReward(player), actions)
    if np.random.rand() < epsilon:
        action = actions[np.random.randint(len(actions))]
        agent.s = None
    game.makeMove(player, action)


np.set_printoptions(suppress=True, precision=8)

numTrain = 5000
numRepeatGames = 1000
trainBoardSize = 6
trainShips = [2, 3, 4]

numPlay = 100
playBoardSize = trainBoardSize#10
playShips = trainShips#[2, 3, 3, 4, 5]

g = BattleshipGame(trainBoardSize, trainShips)
agent1 = QFunctionApproximator(1, g.getNumFeatures(), batchSize=1000, gamma=1, decay=0.95, alpha=0.1, minWeight=0, maxWeight=0)

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