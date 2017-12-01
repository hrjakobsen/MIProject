from unittest import TestCase

import numpy as np

from project.games.pong import PongGame, makeMove, updateBall, NOTHING


class pongTest(TestCase):

    def testCalculateFeaturesReturnsCorrectlySizedArray(self):
        # Arrange
        pongGame = PongGame()
        expectedSize = 2
        state = PongGame()
        player = 1
        action = NOTHING

        # Act
        result = pongGame.calculateFeatures(state, player, action).size

        # Assert
        self.assertEqual(expectedSize, result)

    def testCalculateFeaturesReturnsNotNoneObjects(self):
        # Arrange
        pongGame = PongGame()
        state = PongGame()
        player = 1
        action = NOTHING

        # Act
        result = pongGame.calculateFeatures(state, player, action)

        # Assert
        for i in result:
            self.assertIsNotNone(i)

    def testUpdateBallIfBallBouncesOnBottomWall(self):
        # Arrange
        ballVelocity = np.array([0.5, 0.8])
        ballPosition = np.array([50, 48])
        pongGame = PongGame(ballVelocity, ballPosition)
        expectedVelocity = np.array([0.5, -0.8])

        # Act
        result = updateBall(pongGame)

        # Assert
        self.assertEqual(expectedVelocity[1], result[1][1])

    def testUpdateBallIfBallBouncesOnTopWall(self):
        # Arrange
        ballVelocity = np.array([0.5, -0.8])
        ballPosition = np.array([50, 2])
        pongGame = PongGame(ballVelocity, ballPosition)
        expectedVelocity = np.array([0.5, 0.8])

        # Act
        result = updateBall(pongGame)

        # Assert
        self.assertEqual(expectedVelocity[1], result[1][1])

    def testUpdateBallBounceOnRightPaddle(self):
        # Arrange
        ballVelocity = np.array([0.9, 0.1])
        ballPosition = np.array([98, 25])
        pongGame = PongGame(ballVelocity, ballPosition)
        expectedVelocity = np.array([-0.9, 0.1])

        # Act
        result = updateBall(pongGame)

        # Assert
        self.assertEqual(expectedVelocity[1], result[1][1])

    def testUpdateBallBounceOnLeftPaddle(self):
        # Arrange
        ballVelocity = np.array([-0.9, 0.1])
        ballPosition = np.array([2, 25])
        pongGame = PongGame(ballVelocity, ballPosition)
        expectedVelocity = np.array([0.9, 0.1])

        # Act
        result = updateBall(pongGame)

        # Assert
        self.assertEqual(expectedVelocity[1], result[1][1])

    def testUpdateBallIfInCorner(self):
        # Arrange
        ballPosition = np.array([100, 47.5])
        ballVelocity = np.array([0.7, 0.7])
        pongGame = PongGame(ballVelocity, ballPosition)
        expectedVelocity = np.array([-0.7, -0.7])

        # Act
        result = updateBall(pongGame)

        # Assert
        self.assertEqual(expectedVelocity, result[1][1])

    def testUpdateBall(self):
        # Arrange
        ballPosition = np.array([0, 47.5])
        ballVelocity = np.array([0.3, 0.3])
        pongGame = PongGame(ballVelocity, ballPosition)
        expected = np.array([-0.3, -0.3])

        # Act
        result = updateBall(pongGame)

        # Assert
        self.assertEqual(expected, result[1][1])

    def