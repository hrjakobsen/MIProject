import time
import numpy as np
from agents.AgentRandom import RandomAgent
from agents.HexagonFunctionApproximator import QFunctionApproximator
from games.pong import PongGame, makeMove

np.set_printoptions(suppress=True, precision=2)
np.random.seed(0)

def learn(agent1, agent2, numGames, epsilon):
    p1Wins = 0
    interval = numGames / 100

    for x in range(numGames):
        game = PongGame()

        while not game.gameEnded():
            action1 = getMove(agent1, game, 1, epsilon)
            action2 = getMove(agent2, game, 2, epsilon)

            game = makeMove(game, action1, action2)

            if game.gameEnded():
                break

        agent1.finalize(game, game.getReward(1))
        agent2.finalize(game, game.getReward(2))

        p1Wins += 1 if game.getReward(1) == 1 else 0
        if x % interval == 0:
            print("\rPlayed %s/%s games" % (x, numGames), end="")

    return p1Wins


def getMove(agent, game: PongGame, player, epsilon):
    action = agent.getMove(game, game.getReward(player))
    if np.random.rand() < epsilon:
        action = np.random.randint(3)
        agent.s = None
    return action

g = PongGame()
features = len(g.getFeatures(1))
agent1 = QFunctionApproximator(1, features, g.getActions(), gamma=1, batchSize=50)
agent2 = QFunctionApproximator(1, features, g.getActions(), gamma=1, batchSize=50)
numGames = 10

startTime = time.time()
wins = learn(agent1, agent2, numGames, 0.1)

print("\nDone! - Played {0} games. Took {1}s. Won {2} games.".format(str(numGames), str(
    round(time.time() - startTime, 2)), str(wins)))