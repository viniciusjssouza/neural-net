import numpy as np
import somoclu
from matplotlib import pyplot

from som.kohonen import Training, KohonenMap


class Experiment:
    colors = ['r', 'g', 'b']

    def __init__(self, data, map, learning_strategy, changes):
        self.data = data
        self.learning_strategy = learning_strategy
        self.changes = changes
        self.map = map

    def run_all(self):
        for attr, attr_values in self.changes.items():
            self.run_single(attr, attr_values)
        # run for the input combination
        listener = self.run_training()
        plot_options = {"color": 'r', "label": "Best combination"}
        self.plot_distances(listener, plot_options)
        pyplot.show()
        self.show_umatrix();

    def run_single(self, attr, attr_values):
        for i in range(0, len(attr_values)):
            curr_val = getattr(self.learning_strategy, attr)
            setattr(self.learning_strategy, attr, attr_values[i])
            listener = self.run_training()
            setattr(self.learning_strategy, attr, curr_val)
            plot_options = {"color": Experiment.colors[i], "label": "{} = {}".format(attr, attr_values[i])}
            self.plot_distances(listener, plot_options)
        pyplot.show()

    def run_training(self):
        listener = TrainingListener()
        self.map.reset()
        training = Training(self.map, self.data.values, self.learning_strategy, listener)
        training.run()
        return listener

    def show_umatrix(self):
        som = somoclu.Somoclu(self.map.cols, self.map.rows)
        som.train(self.data.values)
        som.view_umatrix(bestmatches=True)

    def plot_distances(self, listener, plot_options):
        pyplot.figure("distance")
        pyplot.plot(listener.distances, "{}-+".format(plot_options["color"]),
                    linewidth=1, label="{}".format(plot_options["label"]))
        pyplot.legend(loc='best')
        pyplot.xlabel('Epochs')


class TrainingListener:

    def __init__(self):
        self.distances = []

    def before_epoch(self, epoch, map):
        print("======= Epoch {} =======".format(epoch))

    def after_epoch(self, epoch, distances):
        self.distances.append(np.mean(distances))
