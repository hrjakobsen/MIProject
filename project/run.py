from games.HexagonGame import HexagonGame
from agents.HexagonQLearner import HexLearner
from agents.HexagonRandom import HexRandom
import numpy as np
import time
import copy
import os.path

def learn(agent1, agent2, numGames, epsilon):
    p2Start = False
    p1Wins = 0
    interval = numGames / 100

    startGame = HexagonGame(5, 5)

    for x in range(numGames):
        game = HexagonGame(3, 3)
        #game = copy.deepcopy(startGame)

        if p2Start:
            makeMove(agent2, game, 2, epsilon)

        while not game.gameEnded():
            makeMove(agent1, game, 1, epsilon)
            if game.gameEnded():
                break

            makeMove(agent2, game, 2, epsilon)


        agent1.finalize(game, game.getReward(1))
        agent2.finalize(game, game.getReward(2))
        
        p2Start = not p2Start
        p1Wins += 1 if game.getReward(1) == 1 else 0
        if (x % interval == 0):
            print("\rPlayed %s/%s games" % (x, numGames), end="")

    return p1Wins


def makeMove(agent, game, player, epsilon):
    action = agent.getMove(game, game.getReward(player))
    if np.random.rand() < epsilon:
        action = np.random.randint(5)
    game.makeMove(player, action)

np.random.seed(0)

agent1 = HexLearner(1, {}, {})
agent2 = HexLearner(2, {}, {})

numGames = 1000
startTime = time.time()

wins = learn(agent1, agent2, numGames, 0.1)

#agent1.save()

print("\nDone! - Played {0} games. Took {1}s. Saw {2} states. Won {3} games.".format(str(numGames), str(
    round(time.time() - startTime, 2)), str(len(agent1.Q)), str(wins)))