import numpy as np
from unittest import TestCase
from games.battleshipSingle import BattleshipGame
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

    def testShootingWaterUpdatesBoard(self):
        game = deepcopy(self.initialGame)
        waterValue = 2

        game.makeMove(1, (0, 0))

        self.assertEqual(waterValue, game.p2Game.board[0, 0])

    def testShootingShipUpdatesBoard(self):
        game = deepcopy(self.initialGame)
        shipValue = 3

        game.makeMove(1, (0, 2))

        self.assertEqual(shipValue, game.p2Game.board[0, 2])

    def testShootingLastShipEndsGame(self):
        game = deepcopy(self.initialGame)

        shipLocations = [
            (0, 2),
            (0, 3),
            (0, 4),

            (1, 3),
            (1, 4),
            (1, 5),

            (5, 6),
            (5, 7),

            (1, 1),
            (2, 1),
            (3, 1),
            (4, 1),

            (2, 0),
            (3, 0),
            (4, 0),
            (5, 0),
            (6, 0)
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