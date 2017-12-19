import numpy as np
from unittest import TestCase
from games.battleship import *
from copy import deepcopy


class BattleShipTests(TestCase):
    def setUp(self):
        np.random.seed(0)
        self.initialGame = Battleship()
        """
            Player 1 board is:
            [[0 1 0 1 0 0 0 0 0 0]
             [0 1 0 1 0 0 0 0 0 0]
             [0 1 0 1 0 0 0 0 0 0]
             [0 0 0 1 0 0 0 0 0 0]
             [0 0 0 1 0 0 0 1 0 0]
             [0 0 0 0 0 0 0 1 0 0]
             [0 0 0 0 0 0 0 1 1 0]
             [0 0 0 0 0 0 0 1 1 0]
             [0 0 0 0 0 0 0 0 1 0]
             [0 0 0 0 0 1 1 0 0 0]]
             
            Player 2 board is:
            [[0 1 0 0 0 0 0 1 1 0]
             [0 1 0 0 0 0 0 0 0 0]
             [0 1 0 0 0 0 0 0 0 0]
             [0 0 0 1 1 1 0 0 0 0]
             [0 0 0 1 1 1 1 1 0 0]
             [1 1 1 1 0 0 0 0 0 0]
             [0 0 0 0 0 0 0 0 0 0]
             [0 0 0 0 0 0 0 0 0 0]
             [0 0 0 0 0 0 0 0 0 0]
             [0 0 0 0 0 0 0 0 0 0]]
        """

    def testShootingWaterUpdatesBoard(self):
        game = deepcopy(self.initialGame)

        game.makeMove(1, (0, 0))

        self.assertEqual(WATERHIT, game.p2Game.board[0, 0])

    def testShootingShipUpdatesBoard(self):
        game = deepcopy(self.initialGame)

        game.makeMove(1, (0, 1))

        self.assertEqual(SHIPHIT, game.p2Game.board[0, 1])

    def testShootingLastShipEndsGame(self):
        game = deepcopy(self.initialGame)

        shipLocations = [
            (0, 1),
            (1, 1),
            (2, 1),

            (0, 7),
            (0, 8),

            (3, 3),
            (3, 4),
            (3, 5),

            (4, 3),
            (4, 4),
            (4, 5),
            (4, 6),
            (4, 7),

            (5, 0),
            (5, 1),
            (5, 2),
            (5, 3),
        ]

        for loc in shipLocations:
            game.makeMove(1, loc)

        self.assertTrue(game.gameEnded())

    def testMoveKeepsGameGoing(self):
        game = deepcopy(self.initialGame)

        game.makeMove(1, (0, 0))
        game.makeMove(2, (0, 0))

        self.assertFalse(game.gameEnded())

    def testCannotPlayActionTwice(self):
        game = deepcopy(self.initialGame)

        self.assertIn((0, 0), game.getActions(1))

        game.makeMove(1, (0, 0))

        self.assertNotIn((0, 0), game.getActions(1))