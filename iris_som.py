import pandas as pd
from matplotlib import pyplot
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from som.kohonen import KohonenMap, ExponentialLearning, Training

scaler = MinMaxScaler()
training = None
ROWS = 10
COLS = 10
MAX_ITERATIONS = 10
INITIAL_NEIGHBOURHOOD = 5
INITIAL_LEARNING_RATE = 5


class TrainingListener:

    def before_epoch(self, epoch, map):
        print("======= Epoch {} =======".format(epoch))

    def after_epoch(self, epoch, distances, winner_pos):
        pass
        print("Winner: {}".format(winner_pos))


# ==========================================================================
# Helper Functions
# ==========================================================================
def run_model(train, test):
    global training

    map = KohonenMap(rows=10, cols=10, num_features=4)
    learning_strategy = ExponentialLearning(
        map=map,
        max_iterations=MAX_ITERATIONS,
        initial_neighbourhood_size=INITIAL_NEIGHBOURHOOD,
        initial_learning_rate=INITIAL_LEARNING_RATE
    )
    training = Training(map, train.values, learning_strategy, TrainingListener())
    training.run()


def get_input(dataset):
    inputs = scaler.transform(dataset)
    return [row[:4] for row in inputs]


def get_output(dataset):
    return [[row['virginica'], row['versicolor'], row['setosa']] for index, row in dataset.iterrows()]


def run_training(network, dataset, hyperparameters):
    global training

    inputs, expected_outputs = get_input(dataset), get_output(dataset)
    if training is None:
        training = Training(network=network, inputs=inputs, expected_outputs=expected_outputs,
                            hyperparameters=hyperparameters)
    training.run()
    return training


def get_plot_options(hyperparameters):
    if hyperparameters.momentum is not None:
        return {'color': 'b', 'label': 'Iris com termo de Momentum'}
    elif hyperparameters.regularization_parameter is not None:
        return {'color': 'g', 'label': 'Iris com regularização'}
    else:
        return {'color': 'r', 'label': 'Iris'}


def validate(network, dataset):
    inputs, expected_outputs = get_input(dataset), get_output(dataset)
    success = 0
    for i in range(0, len(inputs)):
        network_output = network.feed_forward(inputs[i])
        max_val = max(network_output)
        output = [1 if val == max_val else 0 for val in network_output]
        print("Output: {}    Expected Output: {}".format(output, expected_outputs[i]))
        if output[0] == expected_outputs[i][0]:
            success += 1
    return success * 100 / len(inputs)


# ==========================================================================
# Main script
# ==========================================================================

data = pd.read_csv('datasets/iris.csv')

# remove class column
data = data.loc[:, data.columns != 'class']

# scale dataset input
scaler.fit(data)

# split dataset into training and validation
train, test = train_test_split(data, test_size=0.2)


run_model(train, test)

