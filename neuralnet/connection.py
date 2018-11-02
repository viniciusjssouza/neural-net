import random

import numpy as np


class Connection:

    def __init__(self, source, sink, weight=None):
        if sink is None:
            raise ValueError("The connection sink was not provided")

        self.source = source
        self.sink = sink
        self.weight = weight or np.random.randn()

    def weighted_signal(self):
        return self.signal() * self.weight

    def signal(self):
        """
        For connections between neurons of the hidden layer, the signal is the output of the source neuron.
        """
        return self.source.output
