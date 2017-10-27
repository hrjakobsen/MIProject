from games.HexagonGame import HexagonGame
from agents.HexagonQLearner import HexLearner
import numpy as np
import time

np.random.seed(0)

NUMTRIALS = 2000
EPSILON = 0.1
latestQ1 = {}
latestN1 = {}
latestQ2 = {}
latestN2 = {}
agent1, agent2 = None, None

startTime = time.time()
startGame = HexagonGame(3, 3)

for x in range(NUMTRIALS):
    agent1 = HexLearner(1, latestQ1, latestN1)
    agent2 = HexLearner(2, latestQ2, latestN2)
    game = HexagonGame(3, 3)
    while not game.gameEnded():
        num = np.random.rand()
        action = agent1.getMove(game, game.getReward(1))
        if num < EPSILON:
            action = np.random.randint(5)
        game.makeMove(1, action)

        action = agent2.getMove(game, game.getReward(2))
        if num < EPSILON:
            action = np.random.randint(5)
        game.makeMove(2, action)

    agent1.getMove(game, game.getReward(1))
    agent1.getMove(game, game.getReward(1))
    agent2.getMove(game, game.getReward(2))
    latestQ1 = agent1.Q
    latestN1 = agent1.N
    latestQ2 = agent2.Q
    latestN2 = agent2.N
    interval = NUMTRIALS / 100
    if x % interval == 0:
        print(f"\rPlayed {x}/{NUMTRIALS} games", end="")

agent1.save()
agent2.save()

print("\nDone! - Played " + str(NUMTRIALS) + " games. Took " + str(round(time.time() - startTime, 2)) + "s. Saw " + str(
    len(latestQ1)) + " states")