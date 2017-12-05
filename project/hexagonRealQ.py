from agents.hexagonBruteforce import HexagonBruteforce
from games.hexagon import HexagonGame
import numpy as np

np.random.seed(2)

game = HexagonGame(5, 5)
agent = HexagonBruteforce(game, 1, gamma=1)

agent.save()
