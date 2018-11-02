import numpy as np


class Sigmoid:

    @staticmethod
    def apply(x):
        return 1.0 / (1.0 + np.exp(x))

    @staticmethod
    def first_derivative(x):
        return Sigmoid.apply(x) * (1 - Sigmoid.apply(x))




