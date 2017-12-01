import unittest
from neuralnetwork.NeuralNetwork import *
import numpy as np


class TestNeuralNetwork(unittest.TestCase):

    def test_inputLayer_returns_linear(self):
        layer = InputLayer(4)
        vals = np.asarray([1, 2, 3, 4])
        output = np.asarray([1, 2, 3, 4])
        np.testing.assert_array_equal(output, layer.calc(vals))

    def test_neuralNetwork_weights_initially_zero(self):
        network = NeuralNetwork([
            InputLayer(10),
            DenseLayer(5, activation='linear'),
            OutputLayer(1, activation='linear')
        ], batch_size=50)
        inputs = np.asarray([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        outputs = np.asarray([0])
        np.testing.assert_array_equal(network.predict(inputs), outputs)


if __name__ == '__main__':
    unittest.main()