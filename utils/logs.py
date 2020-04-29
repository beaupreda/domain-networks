"""
functions to save information about some executions (args, results, ...)

author: David-Alexandre Beaupre
date: 2020-04-28
"""

import os
from argparse import Namespace

import torch


class Logs:
    def __init__(self, savepath: str):
        """
        represents information that we want to save.
        :param savepath: path the log folder of the current run,
        """
        self.savepath = savepath
        self._create_directories()

    def save_args(self, args: Namespace, model: torch.nn.Module, optimizer: torch.optim,
                  criterion: torch.nn.Module) -> None:
        """
        writes to file all relevant information for the execution of a given network.
        :param args: user arguments of the program.
        :param model: model used (from models folder).
        :param optimizer: optimizer used (Adam, Adagrad, SGD, etc.).
        :param criterion: loss function used.
        :return: void
        """
        with open(os.path.join(self.savepath, 'summary.log'), 'w') as file:
            file.write('************ ARGUMENTS ************\n\n')
            for arg in vars(args):
                file.write(f'{arg} = {getattr(args, arg)}\n')
            file.write('\n')
            file.write('************ MODEL ************\n\n')
            file.write(f'{str(model)}\n')
            file.write('\n')
            file.write('************ OPTIMIZER ************\n\n')
            file.write(f'{str(optimizer)}\n')
            file.write('\n')
            file.write('************ CRITERION ************\n\n')
            file.write(f'{str(criterion)}\n')
            file.write('\n')

    def save_accuracy_loss(self, epoch: int, train_accuracy: float, validation_accuracy: float,
                           train_loss: float, validation_loss: float) -> None:
        """
        writes to file (at every epoch) the training/validation accuracy/loss.
        :param epoch: current epoch.
        :param train_accuracy: training accuracy for a given epoch.
        :param validation_accuracy: validation accuracy for a given epoch.
        :param train_loss: training loss for a given epoch.
        :param validation_loss: validation loss for a given epoch.
        :return: void
        """
        with open(os.path.join(self.savepath, 'results.log'), 'a') as file:
            file.write(f'epoch = {epoch}\t'
                       f'train accuracy = {train_accuracy * 100.0:.2f}\t'
                       f'validation accuracy = {validation_accuracy * 100.0:.2f}\t'
                       f'train loss = {train_loss:.2f}\t'
                       f'validation loss = {validation_loss:.2f}\n')

    def _create_directories(self) -> None:
        """
        creates relevant directories (current run and parameters).
        :return: void
        """
        if not os.path.isdir(self.savepath):
            os.mkdir(self.savepath)
        parameters = os.path.join(self.savepath, 'parameters')
        if not os.path.isdir(parameters):
            os.mkdir(parameters)
