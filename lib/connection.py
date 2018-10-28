import random


class Connection:

    def __init__(self, source, sink, weight=None):
        self.source = source
        self.sink = sink
        self.weight = weight or random.uniform(-1, 1)

    def weighted_signal(self):
        return self.signal() * self.weight

    def signal(self):
        raise NotImplementedError


class InputConnection(Connection):

    def __init__(self, sink, weight=None):
        if sink is None:
            raise ValueError("The connection sink was not provided")
        super().__init__(None, sink, weight)
        self._signal = 0

    @property
    def signal(self):
        """For input connections, the signal is the input signal"""
        return self._signal


class InnerConnection(Connection):

    def __init__(self, source, sink, weight=None):
        if source is None:
            raise ValueError("The connection source was not provided")
        if sink is None:
            raise ValueError("The connection sink was not provided")
        super().__init__(None, sink, weight)

    def signal(self):
        """
        For connections between neurons of the hidden layer, the signal is the output of the source neuron.
        """
        return self.source.output
