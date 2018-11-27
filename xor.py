from matplotlib import pyplot

from perceptron.neural_network import MultiLayerPerceptron
from perceptron.training import Training, HyperParameters


def run_training(network, hyperparameters):
    training = Training(network=network, inputs=inputs, expected_outputs=expected_outputs,
                        hyperparameters=hyperparameters)

    print("Running training...")
    training.run()
    return training


def validate(inputs, network):
    for curr_input in inputs:
        output = network.feed_forward(curr_input)
        print("Input: {}   Output: {}".format(curr_input, output))


# XOR Network
network1 = MultiLayerPerceptron([2], number_of_inputs=2, number_of_outputs=1)
network2 = MultiLayerPerceptron([2], number_of_inputs=2, number_of_outputs=1)

inputs = [
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1],
]

expected_outputs = [[0], [1], [1], [0]]

hyperparameters = HyperParameters(
    learning_rate=0.5,
    error_tolerance=1e-2,
    max_epochs=20000
)

training = run_training(network1, hyperparameters)
pyplot.plot(training.errors, "r-+", linewidth=1, label="MSE - XOR")
validate(inputs, network1)

# With momentum term
hyperparameters.momentum = 0.2
training = run_training(network2, hyperparameters)
pyplot.plot(training.errors, "b-+", linewidth=1, label="MSE - XOR with Momentum")
pyplot.show()
validate(inputs, network2)

