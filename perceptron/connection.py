import random

import numpy as np


class Connection:

    def __init__(self, source, sink, weight=None):
        if sink is None:
            raise ValueError("The connection sink was not provided")

        self.source = source
        self.sink = sink
        self._weight = weight or np.random.randn()
        self.previous_weight = None

    def weighted_signal(self):
        return self.signal() * self.weight

    def signal(self):
        """
        For connections between neurons of the hidden layer, the signal is the output of the source neuron.
        """
        return self.source.output

    @property
    def weight(self):
        return self._weight

    @weight.setter
    def weight(self, new_weight):
        self.previous_weight = self._weight
        self._weight = new_weight

