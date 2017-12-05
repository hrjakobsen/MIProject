from games.hexagon import HexagonGame
from agents.TabularQLearner import TabularQLearner
from agents.QFunctionApproximator import QFunctionApproximator
from agents.RandomAgent import RandomAgent
from agents.GreedyHexAgent import GreedyHexAgent

import numpy as np
import time
import copy

def learn(agent1, agent2, numGames, epsilon, width=3, height=3):
    p2Start = False
    #p1Wins = 0
    outcomes = []
    interval = numGames / 100

    startGame = HexagonGame(width, height)

    print(startGame.board)

    for x in range(numGames):
        game = copy.deepcopy(startGame)
        #game = HexagonGame(width, height)

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
        #p1Wins += 1 if game.getReward(1) == 1 else 0
        outcomes.append(1 if game.getReward(1) == 1 else 2)
        if x % interval == 0:
            print("\rPlayed %s/%s games" % (x, numGames), end="")

    #return p1Wins
    return outcomes


def makeMove(agent, game, player, epsilon):
    action = agent.getMove(game, game.getReward(player), game.getActions())
    if np.random.rand() < epsilon:
        action = g.getActions()[np.random.randint(5)]
        agent.s = None
    game.makeMove(player, action)


np.set_printoptions(suppress=True, precision=2)
np.random.seed(0)

numGames = 50000#0000
width = 3
height = 3

g = HexagonGame(width, height)
agent1 = TabularQLearner.load(1)#TabularQLearner(g.getActions(), {}, {}, 1)
agent2 = RandomAgent()

startTime = time.time()
outcomes = learn(agent1, agent2, numGames, 0.5, width, height)
print("\nDone! - Played {0} games. Took {1}s. Won {2} games.".format(str(numGames), str(
    round(time.time() - startTime, 2)), str(len([g for g in outcomes if g == 1]))))

#print(outcomes)

p1Wins = []
wonGames = 0
for i in range(len(outcomes)):
    if outcomes[i] == 1:
        wonGames += 1
    p1Wins.append(wonGames)

#print(p1Wins)

agent1.save(1)
print(len(agent1.Q))

#plt.plot(p1Wins)
#plt.show()

#num_bins = numGames // 100
#n, bins, patches = ax.hist(p1Wins, num_bins, normed=0, histtype='step', cumulative=True)
#n, bins, patches = ax.hist(p1Wins, num_bins, normed=0)

