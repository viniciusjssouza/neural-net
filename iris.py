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


# ==========================================================================
# Helper Functions
# ==========================================================================
def run_model(train, test, hyperparameters, training_iterations=1):
    global training

    network = MultiLayerPerceptron([6], number_of_inputs=4, number_of_outputs=3)
    accuracies = []
    training = None
    for it in range(0, training_iterations):
        run_training(network, train, hyperparameters)
        accuracies.append(validate(network, train))

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

# convert classes to numeric values
data['virginica'] = 0
data['versicolor'] = 0
data['setosa'] = 0
data.loc[data['class'] == 'Iris-virginica', 'virginica'] = 1
data.loc[data['class'] == 'Iris-versicolor', 'versicolor'] = 1
data.loc[data['class'] == 'Iris-setosa', 'setosa'] = 1

# remove class column
data = data.loc[:, data.columns != 'class']

# scale dataset input
scaler.fit(data)

# split dataset into training and validation
train, test = train_test_split(data, test_size=0.2)

hyperparameters = HyperParameters(
    learning_rate=1,
    error_tolerance=0.01,
    max_epochs=10
)

run_model(train, test, hyperparameters, training_iterations=10)

# run again with momentum term
hyperparameters.momentum = 0.8
run_model(train, test, hyperparameters, training_iterations=10)

hyperparameters.momentum = None
hyperparameters.regularization_parameter = 1
run_model(train, test, hyperparameters, training_iterations=10)

pyplot.figure(MSE_PLOT)
pyplot.show()

pyplot.figure(ACCURACY_PLOT)
pyplot.show()
