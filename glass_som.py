import numpy as np
import pandas as pd
import somoclu
from matplotlib import pyplot
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from som.kohonen import KohonenMap, ExponentialLearning, Training, LinearLearning

scaler = MinMaxScaler()
training = None
ROWS = 10
COLS = 10
MAX_ITERATIONS = 20
NUM_FEATURES = 9


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


def run_training(map, learning_strategy):
    listener = TrainingListener()
    training = Training(map, train.values, learning_strategy, listener)
    training.run()
    plot_distances(listener)
    som = somoclu.Somoclu(COLS, ROWS)
    som.train(train.values)
    som.view_umatrix(bestmatches=True)


# ==========================================================================
# Main script
# ==========================================================================

data = pd.read_csv('datasets/glass.csv')

# remove class and Id column
data = data.loc[:, data.columns != 'Type']
data = data.loc[:, data.columns != 'Id']

# scale dataset input
scaler.fit(data)

# split dataset into training and validation
train, test = train_test_split(data, test_size=0.2)

## Exponential
map = KohonenMap(rows=ROWS, cols=COLS, num_features=NUM_FEATURES)
exponential_learning = ExponentialLearning(
    map=map,
    learning_decay=0.25 * MAX_ITERATIONS,
    neighbourhood_decay=0.25 * MAX_ITERATIONS,
    max_iterations=MAX_ITERATIONS,
    initial_neighbourhood_size=5,
    initial_learning_rate=0.2
)
run_training(map, exponential_learning)

## Linear
map = KohonenMap(rows=ROWS, cols=COLS, num_features=NUM_FEATURES)
linear_learning = LinearLearning(
    map=map,
    initial_learning_rate=0.001,
    neighbourhood_radius=4,
    max_iterations=MAX_ITERATIONS
)
run_training(map, linear_learning)