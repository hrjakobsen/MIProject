import numpy as np
import math


class Node(object):
    def __init__(self):
        self.batchSize = None
        self.weights = None
        self.a = 0

    def calculate(self, inputs):
        result = np.dot(np.asarray(inputs), self.weights)
        result_activated = self.activation_function(result)
        self.a = self.a + result_activated / self.batchSize
        return result_activated

    def activation_function(self, res):
        pass

    def derivative(self, res):
        pass

    def build(self, weights, batch_size):
        self.weights = np.zeros(weights)
        self.batchSize = batch_size

    def updateWeight(self):
        pass


class SigmoidNode(Node):
    def activation_function(self, res):
        return 1/(1 + math.e**(-res))

    def derivative(self, res):
        return (self.activation_function(res)-1)/self.activation_function(res)


class LinearNode(Node):
    def activation_function(self, res):
        return res

    def derivative(self, res):
        return 1


class ReLUNode(Node):
    def activation_function(self, res):
        return max(res, 0)

    def derivative(self, res):
        if res < 0:
            return 0
        return 1


class Layer(object):
    def __init__(self):
        self.previousLayer = None
        self.batchSize = None
        self.layer = []
        self.learningRateParameter = 0.05

    def calc(self, res):
        pass

    def build(self, nodes_before, batch_size):
        pass

    def getNumberOfNodes(self):
        pass

    def propagate(self, forward_pass, expected_output):
        pass

    def propagateHidden(self, layer, errors):
        pass

    def finishBatch(self):
        for node in self.layer:
            node.a = 0


class DenseLayer(Layer):
    def __init__(self, nodes, activation):
        super().__init__()
        activation_functions = {
            'sigmoid': SigmoidNode,
            'linear': LinearNode,
            'ReLU': ReLUNode
        }

        cls = activation_functions[activation]

        self.layer = [cls() for _ in range(nodes)]

    def calc(self, input):
        output = np.asarray([node.calculate(input) for node in self.layer])
        return output

    def build(self, previous_layer, batch_size):
        for node in self.layer:
            node.build(previous_layer.getNumberOfNodes(), batch_size)
        self.previousLayer = previous_layer

    def getNumberOfNodes(self):
        return len(self.layer)

    def propagateHidden(self, layer, errors):
        for i, node in enumerate(self.layer):
            node_error = sum([layer.layer[j].weigths[i] * errors[j] for j in range(layer.getNumberOfNodes())])
            for w in len(node.weights):
                node.weights[w] -= (self.previousLayer.layer[w].a * node_error) * self.learningRateParameter


class OutputLayer(DenseLayer):
    def propagate(self, forward_pass, expected_output):
        error = self.partialError(forward_pass, expected_output)
        for i, node in enumerate(self.layer):
            node_error = error * node.derivative(expected_output[i])
            for w in len(node.weights):
                node.weights[w] -= (self.previousLayer.layer[w].a * node_error) * self.learningRateParameter

        self.previousLayer.propagateHidden(self)

    @staticmethod
    def partialError(forward_pass, expected_output):
        n = len(forward_pass)
        error = (1 / n) * sum([(forward_pass[i] - expected_output[i]) * (-expected_output[i]) for i in range(n)])
        return error


class InputLayer(Layer):
    def __init__(self, nodes):
        super().__init__()
        self.nodeCount = nodes

    def build(self, previous_layer, batch_size):
        self.batchSize = batch_size

    def calc(self, inputs):
        if len(inputs) != self.nodeCount:
            raise ValueError('Wrong number of input nodes. Expected {} got {}'.format(self.nodeCount, len(inputs)))
        return inputs

    def getNumberOfNodes(self):
        return self.nodeCount


class NeuralNetwork(object):
    def __init__(self, layers, batch_size):
        if len(layers) == 0:
            raise ValueError("You must specify at least one layer")
        if batch_size < 1:
            raise ValueError("Batch size must be at least one")
        self.layers = layers
        for i in range(1, len(layers)):
            self.layers[i].build(self.layers[i-1], batch_size)

    def predict(self, inputs):
        res = inputs
        for l in self.layers:
            res = l.calc(res)
        return res

    def backpropagate(self, training_inputs, expected_outputs):
        self.layers[-1].propagate(training_inputs, expected_outputs)

    @staticmethod
    def sumOfSquaredErrors(inputs, outputs):
        sse = 0
        for i, inp in enumerate(inputs):
            sse += (inp - outputs[i])**2

        return sse / len(inputs)


