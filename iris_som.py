import numpy as np
import pandas as pd
import somoclu
from matplotlib import pyplot
from sklearn.preprocessing import MinMaxScaler

from som.kohonen import KohonenMap, ExponentialLearning, Training, LinearLearning

scaler = MinMaxScaler()
ROWS = 3
COLS = 3
MAX_ITERATIONS = 50
NUM_FEATURES = 4


class TrainingListener:

    def __init__(self):
        self.distances = []

    def before_epoch(self, epoch, map):
        print("======= Epoch {} =======".format(epoch))

    def after_epoch(self, epoch, distances):
        self.distances.append(np.mean(distances))


# ==========================================================================
# Helper Functions
# ==========================================================================

def plot_distances(listener):
    pyplot.figure("distance")
    pyplot.plot(listener.distances, "b-+", linewidth=1, label="Winner - distances")
    pyplot.legend(loc='best')
    pyplot.xlabel('Epochs')
    pyplot.show()


def run_training(data, map, learning_strategy):
    listener = TrainingListener()
    training = Training(map, data.values, learning_strategy, listener)
    training.run()
    plot_distances(listener)
    som = somoclu.Somoclu(COLS, ROWS)
    som.train(data.values)
    som.view_umatrix(bestmatches=True)


# ==========================================================================
# Main script
# ==========================================================================

data = pd.read_csv('datasets/iris.csv')

# remove class column
data = data.loc[:, data.columns != 'class']

## Exponential
map = KohonenMap(rows=ROWS, cols=COLS, num_features=NUM_FEATURES)
exponential_learning = ExponentialLearning(
    map=map,
    learning_decay=0.2 * MAX_ITERATIONS,
    neighbourhood_decay=0.2 * MAX_ITERATIONS,
    max_iterations=MAX_ITERATIONS,
    initial_neighbourhood_size=3,
    initial_learning_rate=0.1
)
run_training(data, map, exponential_learning)

## Linear
map = KohonenMap(rows=ROWS, cols=COLS, num_features=NUM_FEATURES)
linear_learning = LinearLearning(
    map=map,
    initial_learning_rate=0.001,
    neighbourhood_radius=4,
    max_iterations=MAX_ITERATIONS
)
run_training(data, map, linear_learning)