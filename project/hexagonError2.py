from agents.hexaGridBruteforce import HexaGridBruteforce
from games.hexaGrid import HexaGrid, getHash
from agents.qFunctionTabular import QFunctionTabular
from agents.qFunctionSGD import QFunctionSGD
from agents.random import Random
from agents.hexaGridGreedy import HexaGridGreedy
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

        agent1.finalize(game)
        agent2.finalize(game)

        p2Start = not p2Start
        outcomes.append(1 if game.getReward(1) == 1 else 2)
        if x % interval == 0:
            print("\rPlayed %s/%s games" % (x, numGames), end="")

    return outcomes


def makeMove(agent, game, player, epsilon):
    actions = game.getActions(player)
    action = agent.getMove(game)

    if agent == agent1:
        predictQ = agent.Q.get((game.hash(), action), 0)
        realQ = realQs[game.hash(), action]
        errors.append((predictQ - realQ) ** 2)

    if np.random.rand() < epsilon:
        action = actions[np.random.randint(len(actions))]
        agent.s = None
    game.makeMove(player, action)


np.set_printoptions(suppress=True, precision=2)

numGames = 100000
width = 5
height = 5

agent2 = None

configurations = [
    (
        ((lambda: 0.001), "0.001"),
        1,
        ((lambda val, num: 5 if num < 2 else val), "2")
    ),
    (
        ((lambda: 0.01), "0.01"),
        0.99,
        ((lambda val, num: 5 if num < 5 else val), "20")
    ),
    (
        ((lambda: 1/(100+agent1.N.get((agent1.s.hash(), agent1.a), 0))), "1/(100+x)"),
        0.9,
        ((lambda val, num: 5 if num < 10 else val), "20")
    ),
    (
        ((lambda: 1/(1000+agent1.N.get((agent1.s.hash(), agent1.a), 0))), "1/(1000+x)"),
        0.5,
        ((lambda val, num: 5 if num < 20 else val), "20")
    )
]

np.random.seed(2)
startGame = HexaGrid(width, height)
i = 0
for conf in configurations:
    plt.clf()
    gamma = conf[1]
    fileName = "realQ_{0}x{1}_gamma{2}".format(width, height, gamma)
    if not os.path.exists(fileName):
        print("generating {0}".format(fileName))
        bruteforce = HexaGridBruteforce(startGame, 1, gamma=gamma)
        print("saving {0}".format(fileName))
        bruteforce.save(fileName)
        print("done saving {0}".format(fileName))
    realQs = loadFromFile(fileName)
    learned = 0

    fig = plt

    alpha = conf[0]
    fValue = conf[2]
    alphaF, alphaS = alpha
    errors = []

    agent1 = QFunctionTabular(1, {}, {}, gamma=gamma)

    agent1._alpha = alphaF
    agent1._f = fValue[0]

    agent2 = HexaGridGreedy(2)
    g = copy.deepcopy(startGame)

    np.random.seed(2)
    learn(agent1, agent2, g, numGames, 0.1, width, height)

    learned += numGames

    runningMeanNumber = 100
    numDataPoints = 500

    errors = np.convolve(errors, np.ones((runningMeanNumber,))/runningMeanNumber, mode='valid')
    count = 0

    output = []
    for x in range(0, len(errors), (len(errors)//numDataPoints)):
        print("{0} {1}".format(count, errors[x]))
        output.append(errors[x])
        count += 1

    #plt.xkcd()
    ymax = max(output)
    ymin = min(output)
    plt.ylim([max(0, ymin * 0.9), ymax * 1.1])
    plt.tick_params(
        axis='x',  # changes apply to the x-axis
        which='both',  # both major and minor ticks are affected
        bottom='off',  # ticks along the bottom edge are off
        top='off',  # ticks along the top edge are off
        labelbottom='off')
    fig.scatter(range(len(output)), output)
    fig.suptitle("discount factor={0}, learning rate={1},\nNe={2}, iterations={3}".format(gamma, alphaS, fValue[1], numGames), y=0.99, fontsize=17)

    fig.ylabel('Running MSE')
    fig.xlabel('Moves')

    fileName = "Test{0}".format(i)
    print(fileName)
    fig.savefig("outputs/ErrorPlot{0}.png".format(fileName))

    i += 1
