from neuralnet.neural_network import MultiLayerPerceptron
from neuralnet.training import Training, HyperParameters

# XOR Network
network = MultiLayerPerceptron([2], number_of_inputs=2, number_of_outputs=1)

print(network)

inputs = [
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1],
]

expected_outputs = [[0], [1], [1], [0]]

hyperparameters = HyperParameters(
    learning_rate=0.5,
    error_tolerance=1e-1,
    max_epochs=10,
    max_iterations=1e3
)
training = Training(network=network, inputs=inputs, expected_outputs=expected_outputs, hyperparameters=hyperparameters)

print("Running training...")

training.run()

for curr_input in inputs:
    output = network.feed_forward(curr_input)
    print("Input: {}   Output: {}".format(curr_input, output))
