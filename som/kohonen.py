import numpy as np
from sklearn.metrics.pairwise import euclidean_distances


class KohonenMap:

    def __init__(self, rows, cols, num_features):
        self.weights = np.random.rand(rows, cols, num_features)
        self.rows = rows
        self.cols = cols

    def get_winner_pos(self, distances):
        return np.unravel_index(distances.argmin(), distances.shape)

    def calculate_distances(self, input_data):
        distances = np.zeros([self.rows, self.cols])
        for i in range(0, self.rows):
            for k in range(0, self.cols):
                distances[i][k] = np.linalg.norm(np.subtract(self.weights[i][k], input_data))
        return distances


class ExponentialLearning:
    DEFAULT_NEIGHBORHOOD_SIZE = 3
    DEFAULT_LEARNING_RATE = 0.1

    def __init__(self, map, max_iterations, learning_decay, neighbourhood_decay,
                 initial_neighbourhood_size=DEFAULT_NEIGHBORHOOD_SIZE,
                 initial_learning_rate=DEFAULT_LEARNING_RATE):
        self.map = map
        self.initial_neighbourhood_size = initial_neighbourhood_size
        self.initial_learning_rate = initial_learning_rate
        self.max_iterations = max_iterations
        self.learning_decay = learning_decay
        self.neighbourhood_decay = neighbourhood_decay

    def calc_new_weights(self, epoch, input_data, winner_pos, current_pos):
        learning_rate = self.calc_learning_rate(epoch)
        neighbourhood_size = self.calc_topological_neighbourhood(epoch, winner_pos, current_pos)
        constant_factor = learning_rate * neighbourhood_size
        diff = np.subtract(input_data, self.map.weights[current_pos[0]][current_pos[1]])
        delta = np.multiply(constant_factor, diff)
        self.map.weights[current_pos[0]][current_pos[1]] = np.add(self.map.weights[current_pos[0]][current_pos[1]],
                                                                  delta)

    def calc_learning_rate(self, epoch):
        return self.initial_learning_rate * np.exp(-epoch / self.learning_decay)

    def calc_topological_neighbourhood(self, epoch, winner_pos, current_pos):
        dist = self.calc_distance(winner_pos, current_pos)
        current_neighbourhood_size = self.calc_neighborhood_size(epoch)
        return np.exp(-np.square(dist) / (2 * np.square(current_neighbourhood_size)))

    def calc_distance(self, winner_position, other_position):
        return euclidean_distances([winner_position], [other_position])

    def calc_neighborhood_size(self, epoch):
        return self.initial_neighbourhood_size * np.exp(-epoch / self.neighbourhood_decay)


class LinearLearning:

    def __init__(self, map, max_iterations, neighbourhood_radius, initial_learning_rate):
        self.map = map
        self.initial_learning_rate = initial_learning_rate
        self.max_iterations = max_iterations
        self.neighbourhood_radius = neighbourhood_radius

    def calc_new_weights(self, epoch, input_data, winner_pos, current_pos):
        dist = self.calc_distance(winner_pos, current_pos)
        learning_rate = self.calc_learning_rate(epoch, dist)
        diff = np.subtract(input_data, self.map.weights[current_pos[0]][current_pos[1]])
        delta = np.multiply(learning_rate, diff)
        self.map.weights[current_pos[0]][current_pos[1]] = np.add(self.map.weights[current_pos[0]][current_pos[1]],
                                                                  delta)

    def calc_learning_rate(self, epoch, distance):
        rate = (-epoch / (1.05 * self.max_iterations)) - (distance / self.neighbourhood_radius) + 1
        return self.initial_learning_rate * rate

    def calc_distance(self, winner_position, other_position):
        return euclidean_distances([winner_position], [other_position])


class Training:
    DEFAULT_MAX_ITERATIONS = 1e3

    def __init__(self, map, inputs, learning_strategy, listener=None):
        self.map = map
        self.training_set = inputs
        self.learning_strategy = learning_strategy
        self.listener = listener

    def run(self):
        for epoch in range(1, self.learning_strategy.max_iterations):
            if self.listener:
                self.listener.before_epoch(epoch, map)

            best_distances = []
            for input_data in self.training_set:
                best_distances.append(self.compete(epoch, input_data))

            if self.listener:
                self.listener.after_epoch(epoch, best_distances)

    def compete(self, epoch, input_data):
        distances = self.map.calculate_distances(input_data)
        winner_pos = self.map.get_winner_pos(distances)
        for i in range(0, self.map.rows):
            for k in range(0, self.map.cols):
                self.learning_strategy.calc_new_weights(epoch, input_data, winner_pos, (i, k))
        return distances[winner_pos[0]][winner_pos[1]]
