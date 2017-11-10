import numpy as np
np.random.seed(0)
from games.pong import PongGame, updateBall


game = PongGame()

won = False
while not won:
    coord, vel, who = updateBall(game)
    if who != None:
        print(who)
        won = True
    game.ballPosition, game.ballVelocity = coord, vel