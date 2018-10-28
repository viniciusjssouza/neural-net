from lib.connection import *
from lib.activation_function import sigmoid


class Perceptron:

    def __init__(self, layer=0, activation_function=sigmoid):
        self.output_connections = []
        self.input_connections = []
        self.layer = layer
        self.activation_function = activation_function
        self.output = None

    def connect_to(self, other_perceptron, weight=None):
        if other_perceptron is None:
            raise ValueError('connecting perceptron not provided')

        conn = InnerConnection(source=self, sink=other_perceptron, weight=weight)
        self.output_connections.append(conn)
        other_perceptron.input_connections.append(conn)
        other_perceptron.layer = self.layer + 1

    def run_synapse(self):
        self.output = self.activation_function(self.weighted_sum())
        return self.output

    def weighted_sum(self):
        weighted_signals = map(lambda input: input.weighted_signal(), self.input_connections)
        return sum(weighted_signals)
