import numpy as np


class KohonenMap:

    def __init__(self, rows, cols, num_features):
        self.weights = np.random.rand(rows, cols, num_features)
        self.rows = rows
        self.cols = cols

    def get_winner_pos(self, input_data, distances):
        return np.unravel_index(distances.argmax(), distances.shape)

    def calculate_distances(self, input_data):
        distances = np.zeros(self.rows, self.cols)
        for i in range(0, len(self.rows)):
            for k in range(0, len(self.cols)):
                distances[i][k] = np.linalg.norm(np.subtract(self.weights[i][k], input_data))
        return distances


class ExponentialLearning:
    DEFAULT_NEIGHBORHOOD_SIZE = 3
    DEFAULT_LEARNING_RATE = 0.1

    def __init__(self, map, max_iterations, initial_neighbourhood_size=DEFAULT_NEIGHBORHOOD_SIZE,
                 initial_learning_rate=DEFAULT_LEARNING_RATE):
        self.map = map
        self.initial_neighbourhood_size = initial_neighbourhood_size
        self.initial_learning_rate = initial_learning_rate
        self.max_iterations = max_iterations

    def calc_new_weights(self, epoch, input_data, winner_pos, current_pos):
        learning_rate = self.calc_learning_rate(epoch)
        neighbourhood_size = self.calc_topological_neighbourhood(epoch, winner_pos, current_pos)
        constant_factor = learning_rate * neighbourhood_size
        delta = np.subtract(input_data, self.map.weights[current_pos[0]][current_pos[1]])
        self.map.weights[current_pos[0]][current_pos[1]] = np.multiply(constant_factor, delta)

    def calc_learning_rate(self, epoch):
        return self.initial_learning_rate * np.exp(-epoch / self.max_iterations)

    def calc_topological_neighbourhood(self, epoch, winner_pos, current_pos):
        dist = self.calc_manhatan_distance(winner_pos, current_pos)
        current_neighbourhood_size = self.calc_neighborhood_size(epoch)
        return np.exp(-np.square(dist) / (2 * np.square(current_neighbourhood_size)))

    def calc_manhatan_distance(self, winner_position, other_position):
        dist_vector = np.abs(np.subtract(winner_position, other_position))
        return sum(dist_vector)

    def calc_neighborhood_size(self, epoch):
        return self.initial_neighbourhood_size * np.exp(-epoch / self.max_iterations)


class Training:
    DEFAULT_MAX_ITERATIONS = 1e3

    def __init__(self, map, inputs, learning_strategy, listener=None):
        self.map = map
        self.training_set = inputs
        self.learning_strategy = learning_strategy
        self.listener = listener

    def run(self):
        for epoch in range(0, self.learning_strategy.max_iterations):
            if self.listener:
                self.listener.before_epoch(epoch, map)

            for input_data in self.training_set:
                self.compete(epoch, input_data)

    def compete(self, epoch, input_data):
        distances = self.map.calculate_distances(input_data)
        winner_pos = self.map.get_winner_pos(distances)
        for i in range(0, len(self.map.rows)):
            for k in range(0, len(self.map.cols)):
                self.learning_strategy.calc_new_weights(epoch, input_data, winner_pos, (i, k))
        self.listener.after_epoch(epoch, distances, winner_pos)

    def calculate_distances(self, input_data):
        distances = np.zeros(self.map.rows, self.map.cols)
        for i in range(0, len(self.map.rows)):
            for k in range(0, len(self.map.cols)):
                distances[i][k] = np.linalg.norm(np.subtract(self.map.weights[i][k], input_data))
        return distances
