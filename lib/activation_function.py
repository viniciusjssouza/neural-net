import numpy as np


def sigmoid(weighted_sum):
    return 1.0 / (1.0 + np.exp(weighted_sum))


