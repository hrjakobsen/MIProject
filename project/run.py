from trainer import Trainer
from games.battleship import Battleship
from games.hexaGrid import HexaGrid
from games.pong import Pong

from agents.qFunctionSGD import QFunctionSGD
from agents.hexaGridGreedy import HexaGridGreedy
from agents.battleshipHuntAndTarget import BattleShipHuntAndTarget
from agents.pongGreedy import PongGreedy
from agents.random import Random

import numpy as np

np.set_printoptions(suppress=True, precision=8)
BATTLESHIP = 0
HEXAGRID = 1
PONG = 2

game = HEXAGRID
numTrain = 10
numPlay = 5
numRepeatGames = 100

if game == BATTLESHIP:
    boardSize = 6
    ships = [2, 3, 4]
    gameFunc = lambda: Battleship(boardSize, ships)
    agent2 = BattleShipHuntAndTarget(2, boardSize)

if game == HEXAGRID:
    width = 103
    height = 57
    gameFunc = lambda: HexaGrid(width, height)
    agent2 = HexaGridGreedy(2)

if game == PONG:
    width = 100
    height = 50
    gameFunc = lambda: Pong(width, height)
    agent2 = Random(2)

g = gameFunc()
agent1 = QFunctionSGD(1, g.getNumFeatures(), batchSize=100, gamma=0.9, decay=0.95, alpha=0.001, minWeight=0, maxWeight=0)
#agent2 = QFunctionSGD(2, g.getNumFeatures(), batchSize=100, gamma=1, decay=0.95, alpha=0.001, minWeight=0, maxWeight=0)

trainer = Trainer(agent1, agent2, gameFunc)
trainer.run(numPlay, numTrain, numRepeatGames, verbose=True, visualise=True)
