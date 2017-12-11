from unittest import TestCase

import numpy as np

from project.games.HexagonGame import HexagonGame

class hexagonTest(TestCase):
    def testCalculateFeaturesReturnsCorrectSize(self):
        # Arrange
        hexagonGame = HexagonGame()

        # Act

        # Assert