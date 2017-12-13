from agents.hexagonBruteforce import HexagonBruteforce
from games.hexagon import HexagonGame, getHash
from agents.TabularQLearner import TabularQLearner
from agents.QFunctionApproximator import QFunctionApproximator
from agents.RandomAgent import RandomAgent
from agents.GreedyHexAgent import GreedyHexAgent
import pickle
import os
import numpy as np
import matplotlib.pyplot as plt
import copy


def loadFromFile(fileName):
    with open(fileName, 'rb') as handle:
        return pickle.load(handle)


def learn(agent1, agent2, game, numGames, epsilon, width=3, height=3):
    p2Start = False
    outcomes = []
    interval = numGames / 100

    startGame = copy.deepcopy(game)

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
        predictQ = agent.Q.get((game.hash(), action), 0)
        realQ = realQs[game.hash(), action]
        errors.append((predictQ - realQ) ** 2)

    if np.random.rand() < epsilon:
        action = actions[np.random.randint(len(actions))]
        agent.s = None
    game.makeMove(player, action)


np.set_printoptions(suppress=True, precision=2)

numGames = 150000
width = 5
height = 5

alphas = [#((lambda _: 0.001), "0.001")]
          #((lambda _: 0.01), "0.01")]
          #((lambda x: 1/(100+x)), "1/(100+x)")]
          ((lambda x: 1/(1000+x)), "1/(1000+x)")]
gammas = [0.5]
fValues = [20]

np.random.seed(2)
startGame = HexagonGame(width, height)
i = 17
for gamma in gammas:
    fileName = "realQ_{0}x{1}_gamma{2}".format(width, height, gamma)
    if not os.path.exists(fileName):
        print("generating {0}".format(fileName))
        bruteforce = HexagonBruteforce(startGame, 1, gamma=gamma)
        print("saving {0}".format(fileName))
        bruteforce.save(fileName)
        print("done saving {0}".format(fileName))
    realQs = loadFromFile(fileName)
    learned = 0

    fig = plt


    for alpha in alphas:
        alphaF, alphaS = alpha
        for fValue in fValues:
            errors = []

            agent1 = TabularQLearner({}, {}, gamma=gamma, alphaF=alphaF, fValue=fValue)
            agent2 = GreedyHexAgent(2)
            g = copy.deepcopy(startGame)

            np.random.seed(2)
            learn(agent1, agent2, g, numGames, 0.1, width, height)

            learned += numGames

            runningMeanNumber = 100
            numDataPoints = 1000

            allErrors = errors.copy()

            errors = np.convolve(errors, np.ones((runningMeanNumber,))/runningMeanNumber, mode='valid')
            count = 0

            output = []
            for x in range(0, len(errors), (len(errors)//numDataPoints)):
                print("{0} {1}".format(count, errors[x]))
                output.append(errors[x])
                count += 1

            #plt.xkcd()
            fig.scatter(range(len(output)), output)
            fig.suptitle("gamma={0}, alpha={1}, f={2},\n iterations={3}".format(gamma, alphaS, fValue, numGames), y=0.99, fontsize=17)

            fig.ylabel('Running MSE')
            fig.xlabel('Moves')

            fileName = "Test{0}".format(i)
            print(fileName)
            fig.savefig("Output/ErrorPlot{0}.png".format(fileName))

            fig.show()

            i += 1
            errors = allErrors.copy()
