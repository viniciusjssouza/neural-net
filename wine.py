import pandas as pd
from matplotlib import pyplot
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from neuralnet.neural_network import MultiLayerPerceptron
from neuralnet.training import Training, HyperParameters

scaler = MinMaxScaler()
training = None
MSE_PLOT = 1
ACCURACY_PLOT = 2
MSE_REGULARIZATION = 3
NUMBER_OF_INPUTS = 11
NUMBER_OF_OUTPUTS = 11
MAX_EPOCHS = 200
TRAINING_ITERATIONS = 10
NUMBER_OF_SAMPLES = 500

# extend print area
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

# ==========================================================================
# Helper Functions
# ==========================================================================
def run_model(train, test, hyperparameters, training_iterations=1):
    global training

    network = MultiLayerPerceptron([5], number_of_inputs=NUMBER_OF_INPUTS, number_of_outputs=NUMBER_OF_OUTPUTS)
    accuracies = []
    training = None
    for it in range(0, training_iterations):
        train_sample = train.sample(n=NUMBER_OF_SAMPLES)
        run_training(network, train_sample, hyperparameters)
        accuracies.append(validate(network, train_sample))

    print(validate(network, test))  # print the accuracy on the test set
    plot_mse(hyperparameters)
    plot_accuracy(accuracies, hyperparameters)


def plot_mse(hyperparameters):
    plot_options = get_plot_options(hyperparameters)
    label = "Cost - {}".format(plot_options['label'])
    figure = MSE_PLOT if hyperparameters.regularization_parameter is None else MSE_REGULARIZATION
    pyplot.figure(figure)
    pyplot.plot(training.errors, "{}-+".format(plot_options['color']), linewidth=1, label=label)
    pyplot.legend(loc='best')
    pyplot.xlabel('Epoch')


def plot_accuracy(accuracies, hyperparameters):
    plot_options = get_plot_options(hyperparameters)
    label = "Accuracy - {}".format(plot_options['label'])
    pyplot.figure(ACCURACY_PLOT)
    pyplot.plot(accuracies, "{}-+".format(plot_options['color']), linewidth=1, label=label)
    pyplot.legend(loc='best')
    pyplot.xlabel('Training Session ({} epochs)'.format(hyperparameters.max_epochs))


def get_input(dataset):
    inputs = scaler.transform(dataset)
    return [row[:11] for row in inputs]


def get_output(dataset):
    return [row[11:].values for index, row in dataset.iterrows()]


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
        return {'color': 'b', 'label': 'Vinhos com termo de Momentum'}
    elif hyperparameters.regularization_parameter is not None:
        return {'color': 'g', 'label': 'Vinhos com regularização'}
    else:
        return {'color': 'r', 'label': 'Vinhos'}


def validate(network, dataset):
    inputs, expected_outputs = get_input(dataset), get_output(dataset)
    success = 0
    for i in range(0, len(inputs)):
        network_output = network.feed_forward(inputs[i])
        max_val = max(network_output)
        output = [1 if val == max_val else 0 for val in network_output]
        #print("Output: {}    Expected Output: {}".format(output, expected_outputs[i]))
        if output == expected_outputs[i].tolist():
            success += 1
    return success * 100 / len(inputs)


# ==========================================================================
# Main script
# ==========================================================================

data = pd.read_csv('datasets/winequality-red.csv')

for i in range(11):
    data[str(i)] = 0

for i in range(11):
    data.loc[data['quality'] == i, str(i)] = 1


# remove class column
data = data.loc[:, data.columns != 'quality']

# scale dataset input
scaler.fit(data)

# split dataset into training and validation
train, test = train_test_split(data, test_size=0.2)

hyperparameters = HyperParameters(
    learning_rate=0.25,
    error_tolerance=0.1,
    max_epochs=MAX_EPOCHS
)

hyperparameters.momentum = 0.6
run_model(train, test, hyperparameters, training_iterations=TRAINING_ITERATIONS)

pyplot.figure(MSE_PLOT)
pyplot.show()

pyplot.figure(ACCURACY_PLOT)
pyplot.show()
