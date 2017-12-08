from games.battleshipSingle import BattleshipGame
from agents.QFunctionApproximator import QFunctionApproximator
import numpy as np

np.random.seed(1)
np.set_printoptions(suppress=True, precision=2)

game = BattleshipGame(boardSize=6, ships=[5])
allActions = game.getActions(1)
agent = QFunctionApproximator(1, game.getNumFeatures())

print(game.board)
print()

game.makeMove(1, (0, 0))
print(game.board)
print()

agent.weights = np.array([4.59891631, -9.93408078, 9.08411365, 5.55554139])
featureMap = np.ndarray((game.boardSize, game.boardSize), dtype=np.float64)
featureMap.fill(9)
for action in allActions:
    featureMap[action[0], action[1]] = agent.Q(game, action)#game.calculateFeatures(game, action, None)[3]

print(featureMap)

game.makeMove(1, (4, 4))
print(game.board)
print()
featureMap = np.ndarray((game.boardSize, game.boardSize), dtype=np.float64)
featureMap.fill(9)
for action in allActions:
    featureMap[action[0], action[1]] = agent.Q(game, action)#game.calculateFeatures(game, action, None)[3]

print(featureMap)


