import numpy as np


class Sigmoid:

    @staticmethod
    def apply(x):
        return 1.0 / (1.0 + np.exp(x))

    @staticmethod
    def first_derivative(x):
        return x * (1 - x)


class ReLu:

    @staticmethod
    def apply(x):
        if x > 0:
            return x
        else:
            return 0

    @staticmethod
    def first_derivative(x):
        if x > 0:
            return 1
        else:
            return 0
