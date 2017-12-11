import numpy as np
from unittest import TestCase
from games.battleship import *
from copy import deepcopy

class BattleShipTests(TestCase):
    def setUp(self):
        np.random.seed(0)
        self.initialGame = BattleshipGame()
        """
            Player 1 board is:
            [[0 0 0 0 0 1 0 0 0 0]
             [0 0 0 1 0 1 0 0 0 0]
             [0 0 0 1 0 0 0 0 0 0]
             [0 0 0 1 0 0 0 0 0 0]
             [0 0 1 1 1 1 1 1 0 0]
             [0 0 1 0 0 0 0 0 0 0]
             [0 0 1 0 0 0 0 0 0 0]
             [0 0 0 0 0 0 0 0 0 0]
             [1 1 1 1 0 0 0 0 0 0]
             [0 0 0 0 0 0 0 0 0 0]]
             
             Player 2 board is:
             [[0 0 1 1 1 0 0 0 0 0]
             [0 1 0 1 1 1 0 0 0 0]
             [1 1 0 0 0 0 0 0 0 0]
             [1 1 0 0 0 0 0 0 0 0]
             [1 1 0 0 0 0 0 0 0 0]
             [1 0 0 0 0 0 1 1 0 0]
             [1 0 0 0 0 0 0 0 0 0]
             [0 0 0 0 0 0 0 0 0 0]
             [0 0 0 0 0 0 0 0 0 0]
             [0 0 0 0 0 0 0 0 0 0]]
        """

    def testShootUpdatesBoard(self):
        game = deepcopy(self.initialGame)

        game.makeMove(1, (0, 0))

        self.assertEqual(game.playerOneBoard[0, 0], 2)