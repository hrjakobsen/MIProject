from agents.hexaGridBruteforce import HexaGridBruteforce
from games.hexaGrid import HexaGrid
import numpy as np

np.random.seed(2)

game = HexaGrid(3, 3)
agent = HexaGridBruteforce(game, 1, gamma=1)

agent.save()
