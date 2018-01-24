from trainer import Trainer
from games.battleship import Battleship
from games.hexaGrid import HexaGrid
from games.pong import Pong

from agents.qFunctionNN import QFunctionNN

from agents.qFunctionGD import QFunctionGD
from agents.hexaGridGreedy import HexaGridGreedy
from agents.battleshipHuntAndTarget import BattleShipHuntAndTarget
from agents.pongGreedy import PongGreedy
from agents.random import Random

import numpy as np
np.random.seed(0)

np.set_printoptions(suppress=True, precision=8)
BATTLESHIP = 0
HEXAGRID = 1
PONG = 2

game = BATTLESHIP
numTrain = 20
numPlay = 2
numRepeatGames = 20

if game == BATTLESHIP:
    boardSize = 10
    ships = [2, 3, 3, 4, 5]
    gameFunc = lambda: Battleship(boardSize, ships)

if game == HEXAGRID:
    width = 13
    height = 9
    gameFunc = lambda: HexaGrid(width, height)

if game == PONG:
    width = 100
    height = 50
    gameFunc = lambda: Pong(width, height)

#agent1 = QFunctionGD(1, gameFunc().getNumFeatures(), batchSize=100, gamma=0.9, decay=0.95, alpha=0.001, minWeight=0, maxWeight=0)
#agent2 = QFunctionGD(2, gameFunc().getNumFeatures(), batchSize=100, gamma=0.9, decay=0.95, alpha=0.001, minWeight=0, maxWeight=0)

agent1 = Random(1)
agent2 = QFunctionNN(2, gameFunc().getNumFeatures())

trainer = Trainer(agent1, agent2, gameFunc)
trainer.run(numPlay, numTrain, numRepeatGames, verbose=True, visualise=True)
