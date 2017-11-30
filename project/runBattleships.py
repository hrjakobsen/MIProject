from games.battleshipSingle import BattleshipGame
import numpy as np

np.random.seed(1)

game = BattleshipGame(boardSize=6, ships=[5])
game.getActions(1)

print(game.board)
game.calculateFeatures(game, (0, 0), 1)

game.makeMove(1, (0, 0))
print(game.board)
game.calculateFeatures(game, (4, 0), 1)

game.makeMove(1, (4, 3))
print(game.board)
game.calculateFeatures(game, (4, 0), 1)

game.makeMove(1, (4, 4))
print(game.board)
game.calculateFeatures(game, (4, 0), 1)
game.calculateFeatures(game, (4, 2), 1)
game.calculateFeatures(game, (4, 3), 1)
game.calculateFeatures(game, (4, 5), 1)

game.makeMove(1, (4, 1))
print(game.board)
game.calculateFeatures(game, (4, 2), 1)