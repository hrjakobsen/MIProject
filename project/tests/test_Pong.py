from unittest import TestCase

import numpy as np

from games.pong import Pong, updateBall, UP, DOWN, NOTHING


class pongTest(TestCase):
    def testCalculateFeaturesReturnsCorrectlySizedArray(self):
        pongGame = Pong()
        numberOfFeatures = 3
        player = 1
        action = NOTHING

        result = len(pongGame.getFeatures(player, action))

        self.assertEqual(numberOfFeatures, result)

    def testCalculateFeaturesReturnsNotNoneObjects(self):
        pongGame = Pong()
        player = 1
        action = NOTHING

        result = pongGame.getFeatures(player, action)

        for i in result:
            self.assertIsNotNone(i)

    def testUpdateBallIfBallBouncesOnBottomWall(self):
        ballVelocity = np.array([0.5, 0.8])
        ballPosition = np.array([50, 48])
        pongGame = Pong(ballVelocity, ballPosition)
        expectedVelocity = np.array([0.5, -0.8])

        result = updateBall(pongGame)

        np.testing.assert_array_almost_equal(expectedVelocity, result[1])

    def testUpdateBallIfBallBouncesOnTopWall(self):
        ballVelocity = np.array([0.5, -0.8])
        ballPosition = np.array([50, 2])
        pongGame = Pong(ballVelocity, ballPosition)
        expectedVelocity = np.array([0.5, 0.8])

        result = updateBall(pongGame)

        np.testing.assert_array_almost_equal(expectedVelocity, result[1])

    def testUpdateBallBounceOnRightPaddle(self):
        ballVelocity = np.array([0.9, 0.1])
        ballPosition = np.array([99.5, 25])
        pongGame = Pong(ballVelocity, ballPosition)
        expectedVelocity = np.array([-0.9, 0.1])

        result = updateBall(pongGame)

        np.testing.assert_array_almost_equal(expectedVelocity, result[1])

    def testUpdateBallBounceOnLeftPaddle(self):
        ballVelocity = np.array([-0.9, 0.1])
        ballPosition = np.array([0.5, 25])
        pongGame = Pong(ballVelocity, ballPosition)
        expectedVelocity = np.array([0.9, 0.1])

        result = updateBall(pongGame)

        np.testing.assert_array_almost_equal(expectedVelocity, result[1])

    def testUpdateBallIfInCorner(self):
        ballPosition = np.array([99.5, 49.5])
        ballVelocity = np.array([2, 2])
        pongGame = Pong(ballVelocity, ballPosition)
        pongGame.p2pos = 40
        expectedVelocity = np.array([-2, -2])

        result = updateBall(pongGame)

        np.testing.assert_array_almost_equal(expectedVelocity, result[1])

    def testPlayerOnePositiveRewardIfWon(self):
        ballPosition = np.array([99.5, 2])
        ballVelocity = np.array([2, 0.1])
        pongGame = Pong(ballVelocity, ballPosition)
        expectedReward = 100

        pongGame = pongGame.makeMove(1, NOTHING)
        pongGame = pongGame.makeMove(2, NOTHING)

        self.assertEqual(expectedReward, pongGame.getReward(1))

    def testPlayerTwoNegativeRewardIfLost(self):
        ballPosition = np.array([99.5, 2])
        ballVelocity = np.array([2, 0.1])
        pongGame = Pong(ballVelocity, ballPosition)
        expectedReward = -100

        pongGame = pongGame.makeMove(1, NOTHING)
        pongGame = pongGame.makeMove(2, NOTHING)

        self.assertEqual(expectedReward, pongGame.getReward(2))

    def testPlayerBothZeroRewardIfGameGoing(self):
        pongGame = Pong()

        pongGame.makeMove(1, NOTHING)
        pongGame.makeMove(2, NOTHING)

        self.assertEqual(pongGame.getReward(1), 0)
        self.assertEqual(pongGame.getReward(2), 0)

    def testPlayersMultipleMovesThrowsValueError(self):
        pongGame = Pong()
        pongGame.makeMove(1, UP)
        with self.assertRaises(ValueError):
            pongGame.makeMove(1, UP)

    def testBallMovesAfterTwoMoves(self):
        pongGame = Pong()

        ballPos = pongGame.ballPosition
        ballVelocity = pongGame.ballVelocity

        pongGame = pongGame.makeMove(1, UP)
        pongGame = pongGame.makeMove(2, NOTHING)

        np.testing.assert_array_almost_equal(np.array(
            [ballPos[0] + ballVelocity[0], ballPos[1] + ballVelocity[1]]
        ), pongGame.ballPosition)

    def testBallOneMoveDoesNotAffectBoard(self):
        pongGame = Pong()
        ballPosition = pongGame.ballPosition

        pongGame = pongGame.makeMove(1, NOTHING)

        np.testing.assert_array_almost_equal(ballPosition, pongGame.ballPosition)

    def testGameEndedAfterWinningMove(self):
        ballPosition = np.array([99.5, 2])
        ballVelocity = np.array([2, 0.1])
        pongGame = Pong(ballVelocity, ballPosition)

        pongGame = pongGame.makeMove(1, NOTHING)
        pongGame = pongGame.makeMove(2, NOTHING)

        self.assertTrue(pongGame.gameEnded())

    def testGameNotEndedAfterMoves(self):
        pongGame = Pong()
        pongGame.makeMove(1, NOTHING)
        pongGame.makeMove(2, NOTHING)

        self.assertFalse(pongGame.gameEnded())