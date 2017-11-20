from games.battleship import BattleshipGame

game = BattleshipGame()

game.makeMove(1, (0, 0))

print(game.playerTwoBoard)

print(game.getActions(1))