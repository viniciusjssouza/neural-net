import numpy as np
from sklearn.metrics.pairwise import euclidean_distances


class KohonenMap:

    def __init__(self, rows, cols, num_features):
        self.rows = rows
        self.cols = cols
        self.num_features = num_features
        self.weights = None
        self.reset()

    def get_winner_pos(self, distances):
        return np.unravel_index(distances.argmin(), distances.shape)

    def calculate_distances(self, input_data):
        distances = np.zeros([self.rows, self.cols])
        for i in range(0, self.rows):
            for k in range(0, self.cols):
                distances[i][k] = np.linalg.norm(np.subtract(self.weights[i][k], input_data))
        return distances

    def reset(self):
        self.weights = np.random.rand(self.rows, self.cols, self.num_features)


class BaseLearningFunction:

    def __init__(self, max_iterations, learning_decay, neighbourhood_decay,
                 initial_neighbourhood_size, initial_learning_rate):
        self.initial_neighbourhood_size = initial_neighbourhood_size
        self.initial_learning_rate = initial_learning_rate
        self.max_iterations = max_iterations
        self.learning_decay = learning_decay
        self.neighbourhood_decay = neighbourhood_decay

    def calc_new_weights(self, map, epoch, input_data, winner_pos, current_pos):
        learning_rate = self.calc_learning_rate(epoch)
        neighbourhood_size = self.calc_topological_neighbourhood(epoch, winner_pos, current_pos)
        constant_factor = learning_rate * neighbourhood_size
        diff = np.subtract(input_data, map.weights[current_pos[0]][current_pos[1]])
        delta = np.multiply(constant_factor, diff)
        map.weights[current_pos[0]][current_pos[1]] = np.add(map.weights[current_pos[0]][current_pos[1]], delta)

    def calc_topological_neighbourhood(self, epoch, winner_pos, current_pos):
        dist = self.calc_distance(winner_pos, current_pos)
        current_neighbourhood_size = self.calc_neighborhood_size(epoch)
        return np.exp(-np.square(dist) / (2 * np.square(current_neighbourhood_size)))

    def calc_distance(self, winner_position, other_position):
        return euclidean_distances([winner_position], [other_position])

    def calc_learning_rate(self, epoch):
        raise NotImplementedError("Should be implemented by concrete classes")

    def calc_neighborhood_size(self, epoch):
        raise NotImplementedError("Should be implemented by concrete classes")


class ExponentialLearning(BaseLearningFunction):

    def calc_learning_rate(self, epoch):
        return self.initial_learning_rate * np.exp(-epoch / self.learning_decay)

    def calc_neighborhood_size(self, epoch):
        return self.initial_neighbourhood_size * np.exp(-epoch / self.neighbourhood_decay)


class LinearLearning(BaseLearningFunction):

    def calc_learning_rate(self, epoch):
        rate = (-epoch / (self.learning_decay*self.max_iterations)) + 1
        return self.initial_learning_rate * rate

    def calc_neighborhood_size(self, epoch):
        rate = (-epoch / (self.neighbourhood_decay*self.max_iterations)) + 1
        return self.initial_neighbourhood_size * rate


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
                self.listener.before_epoch(epoch, self.map)

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
                self.learning_strategy.calc_new_weights(self.map, epoch, input_data, winner_pos, (i, k))
        return distances[winner_pos[0]][winner_pos[1]]
