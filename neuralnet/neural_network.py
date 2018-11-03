import numpy as np

from neuralnet.perceptron import Perceptron


class MultiLayerPerceptron:

    def __init__(self, neurons_per_layer, number_of_inputs, number_of_outputs):
        self.layers = []
        self.number_of_inputs = number_of_inputs
        self.number_of_outputs = number_of_outputs
        self._outputs = []

        self.layers.append(self._create_input_layer())
        for layer_level in range(0, len(neurons_per_layer)):
            layer = self._create_layer(layer_level + 1, neurons_per_layer[layer_level])
            self.layers.append(layer)
        self.layers.append(self._create_output_layer())

    def feed_forward(self, inputs):
        if len(inputs) != self.number_of_inputs:
            raise ValueError('Invalid number of inputs. Number of inputs configured: {}'.format(self.number_of_inputs))
        for i in range(0, len(inputs)):
            self.layers[0][i].output = inputs[i]

        for layer in self.layers[1:]:
            for neuron in layer:
                neuron.run_synapse()

        self._outputs = list(map(lambda neuron: neuron.output, self.layers[-1]))
        return self._outputs

    def back_propagate(self, expected_outputs, learning_rate, momentum=None):
        deltas = self.back_propagate_output(expected_outputs, learning_rate, momentum)
        for layer_level in self.backward_hidden_layers():
            deltas = self.back_propagate_hidden_layer(layer_level, deltas, learning_rate, momentum)

    def backward_hidden_layers(self):
        return range(len(self.layers) - 2, 0, -1)

    def back_propagate_output(self, expected_outputs, learning_rate, momentum):
        deltas = []
        for i in range(0, self.number_of_outputs):
            neuron = self.layers[-1][i]
            derivative = neuron.activation_function.first_derivative(self._outputs[i])
            delta = (expected_outputs[i] - neuron.output) * derivative
            for connection in neuron.input_connections:
                delta_w = learning_rate * delta * connection.signal() + self.calculate_momentum(connection, momentum)
                connection.weight = connection.weight - delta_w
            deltas.append(delta)
        return deltas

    def calculate_momentum(self, connection, momentum_factor):
        if momentum_factor:
            prev_weight = connection.previous_weight or connection.weight
            return momentum_factor * (connection.weight - prev_weight)
        else:
            return 0

    def back_propagate_hidden_layer(self, layer_level, next_layer_deltas, learning_rate, momentum):
        deltas = []
        for neuron in self.layers[layer_level]:
            derivative = neuron.activation_function.first_derivative(neuron.output)
            output_connections_weights = [out_conn.weight for out_conn in neuron.output_connections]
            deltas_sum = sum(np.multiply(next_layer_deltas, output_connections_weights))
            delta = derivative * deltas_sum
            for connection in neuron.input_connections:
                delta_w = learning_rate * delta * connection.signal() + self.calculate_momentum(connection, momentum)
                connection.weight = connection.weight + delta_w
            deltas.append(delta)
        return deltas

    def outputs(self):
        return self._outputs

    def _create_input_layer(self):
        input_layer = []
        for i in range(0, self.number_of_inputs):
            perceptron = Perceptron(layer=0)
            input_layer.append(perceptron)
        return input_layer

    def _create_output_layer(self):
        output_layer = []
        for i in range(0, self.number_of_outputs):
            perceptron = Perceptron(layer=len(self.layers))
            output_layer.append(perceptron)
            self._connect_neuron(perceptron)
        return output_layer

    def _create_layer(self, layer_level, number_of_neurons):
        layer = []
        for i in range(0, number_of_neurons):
            perceptron = Perceptron(layer=layer_level)
            layer.append(perceptron)
            self._connect_neuron(perceptron)
        return layer

    def _connect_neuron(self, current_neuron):
        """ Connect the current neuron with all the neurons of the previous layer """
        if current_neuron.layer == 0:
            return
        for previous_layer_neuron in self.layers[current_neuron.layer - 1]:
            previous_layer_neuron.connect_to(current_neuron)

    def __repr__(self):
        return self.__str__()

    def __str__(self) -> str:
        result = ''
        for layer in self.layers:
            result += "========== Layer {}\n".format(layer[0].layer)
            for neuron in layer:
                result += str(neuron) + '\n'
            result += '\n'
        return result
