import numpy as np
from games.hexaGrid import HexaGrid
from unittest import TestCase


initialBoard = np.array(
    [
        [5, -1, 4, -1, 1],
        [3,  2, 1,  4, 3],
        [4,  3, 2,  2, 3],
        [3,  2, 2,  1, 10]
    ]
)


class hexagonTest(TestCase):
    def testColorsTakenWithAction(self):
        game = HexaGrid(5, 3)
        game.board = initialBoard
        expected = np.array(
            [
                [8, -1, 4, -1, 1],
                [8,  2, 1,  4, 3],
                [4,  8, 2,  2, 3],
                [3,  2, 2,  1, 10]
            ]
        )

        game.makeMove(1, 3)

        np.testing.assert_array_equal(expected, game.board)

    def testGameContinuesAfterMove(self):
        game = HexaGrid(5, 3)
        game.board = initialBoard

        game.makeMove(1, 3)

        self.assertFalse(game.gameEnded())

    def testDuplicateMoveDoesNotChangeBoard(self):
        game = HexaGrid(5, 3)
        game.board = initialBoard

        game.makeMove(1, 0)
        game.makeMove(2, 0)
        game.makeMove(1, 0)
        game.makeMove(2, 0)

        np.testing.assert_array_equal(initialBoard, game.board)

    def testGameEndsWhenNoMoves(self):
        game = HexaGrid(3, 3)
        game.board = np.array([
            [5, -1, 1],
            [4, 5, 1],
            [4, 5, 5],
            [4, 4, 10]
        ])

        game.makeMove(1, 4)

        self.assertTrue(game.gameEnded())

    def testRemainingCellsDistributed(self):
        game = HexaGrid(3, 3)
        game.board = np.array([
            [5, -1, 1],
            [4, 5, 1],
            [4, 5, 5],
            [4, 4, 10]
        ])
        expected = np.array([
            [9, -1, 9],
            [9, 9, 9],
            [9, 9, 9],
            [9, 9, 10]
        ])

        game.makeMove(1, 4)

        np.testing.assert_array_equal(expected, game.board)
