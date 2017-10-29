from games.HexagonGame import HexagonGame
from agents.HexagonQLearner import HexLearner
from agents.HexagonRandom import HexRandom
import numpy as np
import time
import copy
import os.path
import matplotlib.pyplot as plt

def learn(agent1, agent2, numGames, epsilon):
    p2Start = False
    p1Wins = 0

    startGame = HexagonGame(5, 5)

    for x in range(numGames):
        #game = HexagonGame(3, 3)
        game = copy.deepcopy(startGame)

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

print("\nDone! - Played " + str(numGames) + " games. Took " + str(round(time.time() - startTime, 2)) + "s. Saw " + str(
    len(agent1.Q)) + " states. Won " + str(wins) + " games.")

'''
NUMTRIALS = 2000
EPSILON = 0.1
latestQ1 = {}
latestN1 = {}
latestQ2 = {}
latestN2 = {}

agent1, agent2 = None, None
wins = 0
startTime = time.time()
p1Start = True
startGame = HexagonGame(3, 3)

for x in range(NUMTRIALS):
    agent1 = HexLearner(1, latestQ1, latestN1)
    #agent2 = HexLearner(2, latestQ2, latestN2)
    agent2 = HexRandom()
    #game = HexagonGame(3, 3)
    game = copy.deepcopy(startGame)

    if not p1Start:
        action = agent2.getMove(game, game.getReward(2))
        if num < EPSILON:
            action = np.random.randint(5)
        game.makeMove(2, action)

    while not game.gameEnded():
        num = np.random.rand()
        action = agent1.getMove(game, game.getReward(1))
        if num < EPSILON:
            action = np.random.randint(5)
        game.makeMove(1, action)

        if game.gameEnded():
            break

        action = agent2.getMove(game, game.getReward(2))
        if num < EPSILON:
            action = np.random.randint(5)
        game.makeMove(2, action)

    agent1.finalize(game, game.getReward(1))
    agent2.finalize(game, game.getReward(2))
    p1Start = not p1Start

    wins += 1 if game.getReward(1) == 1 else 0

    latestQ1 = agent1.Q
    latestN1 = agent1.N
    #latestQ2 = agent2.Q
    #latestN2 = agent2.N
    interval = NUMTRIALS / 100
    if x % interval == 0:
        print("\rPlayed %s/%s games"%(x, NUMTRIALS), end="")

print("\nDone! - Played " + str(NUMTRIALS) + " games. Took " + str(round(time.time() - startTime, 2)) + "s. Saw " + str(
    len(latestQ1)) + " states")

print(wins)
'''