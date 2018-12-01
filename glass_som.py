import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from som.experiment import Experiment
from som.kohonen import KohonenMap, ExponentialLearning, LinearLearning

scaler = MinMaxScaler()
training = None
ROWS = 10
COLS = 10
MAX_ITERATIONS = 10
NUM_FEATURES = 9

# ==========================================================================
# Main script
# ==========================================================================

data = pd.read_csv('datasets/glass.csv')

# remove class and Id column
data = data.loc[:, data.columns != 'Type']
data = data.loc[:, data.columns != 'Id']

## Exponential
exponential_learning = ExponentialLearning(
    learning_decay=5,
    neighbourhood_decay=5,
    max_iterations=MAX_ITERATIONS,
    initial_neighbourhood_size=20,
    initial_learning_rate=0.005
)
changes = {
    'learning_decay': [1, 5, 10],
    'initial_learning_rate': [0.001, 0.0025, 0.005],
    'neighbourhood_decay': [1, 5, 10],
    'initial_neighbourhood_size': [5, 10, 20]
}
map = KohonenMap(rows=ROWS, cols=COLS, num_features=NUM_FEATURES)
experiment = Experiment(data, map, exponential_learning, changes)
experiment.run_all()


## Linear
linearLearning = LinearLearning(
    learning_decay=1.05,
    neighbourhood_decay=1.05,
    max_iterations=MAX_ITERATIONS,
    initial_neighbourhood_size=20,
    initial_learning_rate=0.005
)
changes = {
    'learning_decay': [0.8, 1.05, 1.2],
    'initial_learning_rate': [0.001, 0.005, 0.01],
    'neighbourhood_decay': [0.8, 1.05, 1.2],
    'initial_neighbourhood_size': [5, 10, 20]
}
map = KohonenMap(rows=ROWS, cols=COLS, num_features=NUM_FEATURES)
experiment2 = Experiment(data, map, linearLearning, changes)
experiment2.run_all()

