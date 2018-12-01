import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from som.experiment import Experiment
from som.kohonen import ExponentialLearning, KohonenMap, LinearLearning

scaler = MinMaxScaler()
ROWS = 3
COLS = 3
MAX_ITERATIONS = 10
NUM_FEATURES = 13

# ==========================================================================
# Main script
# ==========================================================================

data = pd.read_csv('datasets/winequality-red.csv')

# remove class column
data = data.loc[:, data.columns != 'type']

## Exponential
exponential_learning = ExponentialLearning(
    learning_decay=5,
    neighbourhood_decay=5,
    max_iterations=MAX_ITERATIONS,
    initial_neighbourhood_size=2,
    initial_learning_rate=0.01
)
changes = {
    'learning_decay': [1, 2, 5],
    'initial_learning_rate': [0.1, 0.05, 0.01],
    'neighbourhood_decay': [1, 2, 5],
    'initial_neighbourhood_size': [2, 4, 8]
}
map = KohonenMap(rows=ROWS, cols=COLS, num_features=NUM_FEATURES)
experiment = Experiment(data, map, exponential_learning, changes)
experiment.run_all()


## Linear
linearLearning = LinearLearning(
    learning_decay=1.05,
    neighbourhood_decay=1.05,
    max_iterations=MAX_ITERATIONS,
    initial_neighbourhood_size=2,
    initial_learning_rate=0.01
)
changes = {
    'learning_decay': [0.8, 1.05, 1.2],
    'initial_learning_rate': [0.01, 0.05, 0.1],
    'neighbourhood_decay': [0.8, 1.05, 1.2],
    'initial_neighbourhood_size': [2, 4, 8]
}
map = KohonenMap(rows=ROWS, cols=COLS, num_features=NUM_FEATURES)
experiment2 = Experiment(data, map, linearLearning, changes)
experiment2.run_all()
