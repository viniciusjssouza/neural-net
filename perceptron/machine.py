import pandas as pd
from matplotlib import pyplot
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from perceptron.neural_network import MultiLayerPerceptron
from perceptron.training import Training, HyperParameters

scaler = MinMaxScaler()
output_scaler = MinMaxScaler()
training = None
MSE_PLOT = 1
RELATIVE_ERROR_PLOT = 2
MSE_REGULARIZATION = 3


# ==========================================================================
# Helper Functions
# ==========================================================================
def run_model(train, test, hyperparameters, training_iterations=1):
    global training

    network = MultiLayerPerceptron([10], number_of_inputs=6, number_of_outputs=1)

    relative_error = []
    training = None
    for it in range(0, training_iterations):
        run_training(network, train, hyperparameters)
        relative_error.append(validate(network, train))
    errors = output_scaler.inverse_transform(relative_error)
    print(output_scaler.inverse_transform([validate(network, test)]))  # print the mean error on the test set
    plot_mse(hyperparameters)
    plot_relative_error(errors, hyperparameters)


def plot_mse(hyperparameters):
    plot_options = get_plot_options(hyperparameters)
    label = "Cost - {}".format(plot_options['label'])
    figure = MSE_PLOT if hyperparameters.regularization_parameter is None else MSE_REGULARIZATION
    pyplot.figure(figure)
    pyplot.plot(training.errors, "{}-+".format(plot_options['color']), linewidth=1, label=label)
    pyplot.legend(loc='best')
    pyplot.xlabel('Epoch')


def plot_relative_error(relative_errors, hyperparameters):
    plot_options = get_plot_options(hyperparameters)
    label = "Relative Error - {}".format(plot_options['label'])
    pyplot.figure(RELATIVE_ERROR_PLOT)
    pyplot.plot(relative_errors, "{}-+".format(plot_options['color']), linewidth=1, label=label)
    pyplot.legend(loc='best')
    pyplot.xlabel('Training Session ({} epochs)'.format(hyperparameters.max_epochs))


def get_input(dataset):
    inputs = scaler.transform(dataset)
    return [row[:6] for row in inputs]


def get_output(dataset):
    outputs = scaler.transform(dataset)
    return [[row[6]] for row in outputs]


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
        return {'color': 'b', 'label': 'Machine Benchmark com termo de Momentum'}
    elif hyperparameters.regularization_parameter is not None:
        return {'color': 'g', 'label': 'Machine Benchmark com regularização'}
    else:
        return {'color': 'r', 'label': 'Machine Benchmark'}


def validate(network, dataset):
    inputs, expected_outputs = get_input(dataset), get_output(dataset)
    outputs = [network.feed_forward(row)[0] for row in inputs]
    expected_outputs = [out[0] for out in expected_outputs]

    relative_errors = []
    for output, expected_output in zip(outputs, expected_outputs):
        print("Output: {}    Expected: {}".format(output, expected_output))
        error = abs(output-expected_output)
        relative_errors.append(error)
    return [pd.np.mean(relative_errors)]

# ==========================================================================
# Main script
# ==========================================================================
COLUMNS_OF_INTEREST = ['myct','mmin','mmax','cach','chmin','chmax','erp']
data = pd.read_csv('datasets/machine.csv')

# convert classes to numeric values
data = data[COLUMNS_OF_INTEREST]

# scale dataset input
scaler.fit(data)
output_scaler.fit(data[["erp"]].values)

# split dataset into training and validation
train, test = train_test_split(data, test_size=0.2)

hyperparameters = HyperParameters(
    learning_rate=5,
    error_tolerance=1e-5,
    max_epochs=20
)
TRAINING_ITERATIONS = 50
run_model(train, test, hyperparameters, training_iterations=TRAINING_ITERATIONS)

# run again with momentum term
hyperparameters.momentum = 0.8
run_model(train, test, hyperparameters, training_iterations=TRAINING_ITERATIONS)

hyperparameters.momentum = None
hyperparameters.regularization_parameter = 1
run_model(train, test, hyperparameters, training_iterations=TRAINING_ITERATIONS)

pyplot.figure(MSE_PLOT)
pyplot.show()

pyplot.figure(RELATIVE_ERROR_PLOT)
pyplot.show()

