"""
functions to make accuracy and loss graphs.

author: David-Alexandre Beaupre
data: 2020-04-28
"""

import os
import matplotlib.figure


class Graph:
    def __init__(self, savepath: str, fig: matplotlib.figure.Figure, ax: matplotlib.figure.Axes, loss: bool = True):
        """
        represents a graph to show the evolution of the loss and accuracy during training.
        :param savepath: path to save the images.
        :param fig: matplotlib figure object.
        :param ax: matplotlib axis object.
        :param loss: whether the graph is a loss one or not.
        """
        self.savepath = savepath
        self.fig = fig
        self.ax = ax
        self.loss = loss

    def create(self, train: float, validation: float) -> None:
        """
        creates the accuracy or validation graph in the axis object.
        :param train: training data (accuracy or loss).
        :param validation: validation data (accuracy or loss).
        :return: void.
        """
        self.ax.plot(train)
        self.ax.plot(validation)
        self.ax.set_xlabel('epochs')
        if self.loss:
            self.ax.set_ylabel('loss')
        else:
            self.ax.set_ylabel('accuracy')
        self.ax.legend(['train', 'validation'])

    def save(self) -> None:
        """
        writes the graph to a PNG file.
        :return: void
        """
        graph = 'loss' if self.loss else 'accuracy'
        save = os.path.join(self.savepath, graph + '.png')
        self.fig.savefig(save)
