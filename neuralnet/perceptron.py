import numpy as np

from neuralnet.connection import *
from neuralnet.activation_function import Sigmoid

import random, string


class Perceptron:

    def __init__(self, layer=0, activation_function=Sigmoid):
        self.output_connections = []
        self.input_connections = []
        self.layer = layer
        self.activation_function = activation_function
        self.output = None
        self.id = self._random_id()
        self.bias = self._create_bias()

    def connect_to(self, other_perceptron, weight=None):
        if other_perceptron is None:
            raise ValueError('connecting perceptron not provided')

        conn = Connection(source=self, sink=other_perceptron, weight=weight)
        self.output_connections.append(conn)
        other_perceptron.input_connections.append(conn)
        other_perceptron.layer = self.layer + 1

    def run_synapse(self):
        self.output = self.activation_function.apply(self.weighted_sum())
        return self.output

    def weighted_sum(self):
        weighted_signals = map(lambda input_signal: input_signal.weighted_signal(), self.input_connections)
        return sum(weighted_signals)

    def _create_bias(self):
        return Bias(self.layer-1, self)

    def _random_id(self):
        letters = string.ascii_lowercase
        random_str = ''.join(random.choice(letters) for i in range(6))
        return 'l{}_{}'.format(self.layer, random_str)

    def __repr__(self):
        return self.__str__()

    def __str__(self) -> str:
        connections = map(lambda conn: '{}(w={:.2f})'.format(conn.sink.id, conn.weight), self.output_connections)
        connections_str = ', '.join(connections)
        return '{} -> bias: {}   output: {}  output_connections: [{}]'.format(self.id, self.bias.output, self.output,
                                                                              connections_str)


class Bias(Perceptron):
    """ A bias is a perceptron with fixed output. It connects only with its "partner" Perceptron"""

    def __init__(self, layer, partner_perceptron):
        self.output_connections = []
        self.input_connections = []
        self.output = np.random.randn()
        self.layer = layer
        self.id = 'l{}_bias_{}'.format(layer, partner_perceptron.id)
        self.connect_to(partner_perceptron)
