import numpy as np


def squared_error(value, expected_value):
    return 0.5 * pow(expected_value - value, 2.0)


class Training:

    def __init__(self, network, inputs, expected_outputs, hyperparameters, cost_function=squared_error):
        self.network = network
        self.training_set = list(zip(inputs, expected_outputs))
        self.current_epoch = None
        self.cost_function = cost_function
        self.hyperparameters = hyperparameters
        self.errors = []

    def run(self):
        self.current_epoch = 1
        self.errors = []
        error = self.hyperparameters.error_tolerance + 1

        while self.unacceptable_error(error) and not self.max_epochs_reached():
            self.log("======================\nEpoch {}", self.current_epoch)
            error = self.run_epoch()
            self.log("Error: {}", error)
            self.errors.append(error)
            self.current_epoch += 1
            # self.log("{}", self.network)
        if self.current_epoch > self.hyperparameters.max_epochs:
            self.log("Training: Max number of epochs reached")

    def run_epoch(self):
        errors = []
        for training_entry in self.training_set:
            inputs, expect_outputs = training_entry
            outputs = self.network.feed_forward(inputs)
            output_vs_expected = zip(outputs, expect_outputs)
            self.log("Input: {}  Expected Output: {}   Output: {}".format(inputs, expect_outputs, outputs))
            cost = map(lambda pair: self.cost_function(pair[0], pair[1]), output_vs_expected)
            error = max(cost)
            errors.append(error)
            if error > self.hyperparameters.error_tolerance:
                self.network.back_propagate(expect_outputs,
                                            learning_rate=self.hyperparameters.learning_rate,
                                            momentum=self.hyperparameters.momentum)
        return np.mean(errors)

    def unacceptable_error(self, current_error):
        return current_error > self.hyperparameters.error_tolerance

    def max_epochs_reached(self):
        return self.current_epoch > self.hyperparameters.max_epochs

    def max_iterations_reached(self, current_iteration):
        return current_iteration > self.hyperparameters.max_iterations

    def log(self, message, params=None):
        if params is None:
            params = []
        print(message.format(params))


class HyperParameters:
    DEFAULT_LEARNING_RATE = 1e-3
    DEFAULT_ERROR_TOLERANCE = 1e-4
    DEFAULT_MAX_EPOCHS = 1e2
    DEFAULT_MAX_ITERATIONS = 1e4

    def __init__(self, learning_rate=DEFAULT_LEARNING_RATE, error_tolerance=DEFAULT_ERROR_TOLERANCE,
                 max_epochs=DEFAULT_MAX_EPOCHS, max_iterations=DEFAULT_MAX_ITERATIONS, momentum=None):
        self.learning_rate = learning_rate
        self.error_tolerance = error_tolerance
        self.max_epochs = max_epochs
        self.max_iterations = max_iterations
        self.momentum = momentum
