"""
training script for the proposed model.

author: David-Alexandre Beaupre
date: 2020-04-29
"""

import os
import argparse
import time as t
import datetime as dt

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim

import utils.misc as misc
from datahandler.LITIV import LITIV
from datahandler.LITIVDataset import TrainLITIVDataset
from models.concatnet import ConcatNet
from models.corrnet import CorrNet
from models.domainnet import DomainNet
from utils.graphs import Graph
from utils.logs import Logs


def training(model: torch.nn.Module, loader: TrainLITIVDataset, criterion: torch.nn.Module, optimizer: torch.optim,
             iterations: int, name: str, cuda: bool, bsize: int) -> (float, float):
    """
    training function to train either the proposed model, or its individual branches.
    :param model: torch model.
    :param loader: data loader.
    :param criterion: torch loss function.
    :param optimizer: torch optimizer.
    :param iterations: number of training iterations.
    :param name: name of the model.
    :param cuda: use GPU or not.
    :param bsize: batch size.
    :return: accuracy and loss values.
    """
    print('training...')
    model.train()
    correct_corr = 0
    correct_concat = 0
    total_loss = 0.0
    for i in range(0, iterations):
        print(f'\r{i + 1} / {iterations}', end='', flush=True)
        rgb, lwir, targets = loader.get_batch()
        if cuda:
            rgb = rgb.cuda()
            lwir = lwir.cuda()
            targets = targets.cuda()
        if name == 'corrnet':
            corr = model(rgb, lwir)
            loss = criterion(corr, targets)
            _, predictions_corr = torch.max(corr, dim=1)
            correct_corr += torch.sum(predictions_corr == targets)
        elif name == 'concatnet':
            concat = model(rgb, lwir)
            loss = criterion(concat, targets)
            _, predictions_concat = torch.max(concat, dim=1)
            correct_concat += torch.sum(predictions_concat == targets)
        else:
            corr, concat = model(rgb, lwir)
            loss = criterion(corr, targets) + criterion(concat, targets)
            _, predictions_corr = torch.max(corr, dim=1)
            _, predictions_concat = torch.max(concat, dim=1)
            correct_corr += torch.sum(predictions_corr == targets)
            correct_concat += torch.sum(predictions_concat == targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss

    accuracy_corr = float(correct_corr) / float(iterations * bsize)
    accuracy_concat = float(correct_concat) / float(iterations * bsize)
    total_loss = float(total_loss) / float(iterations * bsize)

    return max(accuracy_corr, accuracy_concat), total_loss


def validation(model: torch.nn.Module, loader: TrainLITIVDataset, criterion: torch.nn.Module, iterations: int,
               name: str, cuda: bool, bsize: int) -> (float, float):
    """
    validation function to test either the proposed model, or its individual branches.
    :param model: torch model.
    :param loader: data loader.
    :param criterion: torch loss function.
    :param iterations: number of validation iterations.
    :param name: name of the model.
    :param cuda: use GPU or not.
    :param bsize: batch size.
    :return: accuracy and loss values.
    """
    print('validation...')
    model.eval()
    with torch.no_grad():
        correct_corr = 0
        correct_concat = 0
        total_loss = 0.0
        for i in range(0, iterations):
            print(f'\r{i + 1} / {iterations}', end='', flush=True)
            rgb, lwir, targets = loader.get_batch()
            if cuda:
                rgb = rgb.cuda()
                lwir = lwir.cuda()
                targets = targets.cuda()
            if name == 'corrnet':
                corr = model(rgb, lwir)
                loss = criterion(corr, targets)
                _, predictions_corr = torch.max(corr, dim=1)
                correct_corr += torch.sum(predictions_corr == targets)
            elif name == 'concatnet':
                concat = model(rgb, lwir)
                loss = criterion(concat, targets)
                _, predictions_concat = torch.max(concat, dim=1)
                correct_concat += torch.sum(predictions_concat == targets)
            else:
                corr, concat = model(rgb, lwir)
                loss = criterion(corr, targets) + criterion(concat, targets)
                _, predictions_corr = torch.max(corr, dim=1)
                _, predictions_concat = torch.max(concat, dim=1)
                correct_corr += torch.sum(predictions_corr == targets)
                correct_concat += torch.sum(predictions_concat == targets)
            total_loss += loss

    accuracy_corr = float(correct_corr) / float(iterations * bsize)
    accuracy_concat = float(correct_concat) / float(iterations * bsize)
    total_loss = float(total_loss) / float(iterations * bsize)

    return max(accuracy_corr, accuracy_concat), total_loss


def main() -> None:
    torch.manual_seed(42)

    parser = argparse.ArgumentParser()
    parser.add_argument('--save', default='logs', help='folder to save logs from executions')
    parser.add_argument('--fold', type=int, default=1, help='which fold to test on (LITIV dataset only)')
    parser.add_argument('--model', default='concatcorr', help='name of the model to train (filename without the .py)')
    parser.add_argument('--datapath', default='/home/beaupreda/litiv/datasets/litiv')
    parser.add_argument('--loadmodel', default=None, help='name of the model to load, if any')
    parser.add_argument('--max_disparity', type=int, default=121, help='maximum disparity in the dataset')
    parser.add_argument('--patch_size', type=int, default=18, help='half width of the left patch')
    parser.add_argument('--batch_size', type=int, default=64, help='batch size')
    parser.add_argument('--epochs', type=int, default=200, help='number of epochs')
    parser.add_argument('--train_iterations', type=int, default=200, help='number of training iterations per epochs')
    parser.add_argument('--val_iterations', type=int, default=100, help='number of validation iterations per epochs')
    parser.add_argument('--learning_rate', type=float, default=0.01, help='learning rate')
    parser.add_argument('--no_cuda', action='store_true', default=False, help='enables/disables GPU')
    args = parser.parse_args()

    if not os.path.isdir(args.save):
        os.mkdir(args.save)

    now = dt.datetime.now()
    savepath = os.path.join(args.save, now.strftime('%Y%m%d-%H%M%S'))
    logger = Logs(savepath)

    args.cuda = not args.no_cuda and torch.cuda.is_available()

    print('loading dataset...')
    dataset = LITIV(root=args.datapath, psize=args.patch_size, fold=args.fold)
    train_loader = TrainLITIVDataset(dataset.rgb['train'], dataset.lwir['train'], dataset.disp['train'], 'train', args)
    validation_loader = TrainLITIVDataset(dataset.rgb['validation'], dataset.lwir['validation'],
                                          dataset.disp['validation'], 'validation', args)

    print('loading model...')
    if args.model == 'corrnet':
        model = CorrNet(num_channels=3)
    elif args.model == 'concatnet':
        model = ConcatNet(num_channels=3)
    else:
        model = DomainNet(num_channels=3)

    if args.loadmodel is not None:
        parameters = torch.load(args.loadmodel)
        model.load_state_dict(parameters['state_dict'])
    print(f'number of parameters = {misc.get_number_parameters(model)}\n')

    criterion = nn.CrossEntropyLoss(reduction='sum')

    if args.cuda:
        model.cuda()
        criterion.cuda()

    optimizer = optim.Adam(params=model.parameters(), lr=args.learning_rate, weight_decay=0.0005)

    train_losses = []
    validation_losses = []
    train_accuracies = [0.0]
    validation_accuracies = [0.0]
    best_accuracy = 0.0

    for epoch in range(1, args.epochs + 1):
        dawn = t.time()
        print(f'******** EPOCH {epoch} ********')
        if (epoch - 1) % 40 == 0 and (epoch - 1) != 0:
            misc.adjust_learning_rate(optimizer, args)

        train_accuracy, train_loss = training(model, train_loader, criterion, optimizer, args.train_iterations,
                                              args.model, args.cuda, args.batch_size)
        print(f'\ntrain loss = {train_loss:.2f}, train accuracy = {train_accuracy * 100:.2f}')
        validation_accuracy, validation_loss = validation(model, validation_loader, criterion, args.val_iterations,
                                                          args.model, args.cuda, args.batch_size)
        print(f'\nvalidation loss = {validation_loss:.2f}, validation accuracy = {validation_accuracy * 100:.2f}')
        train_losses.append(train_loss)
        validation_losses.append(validation_loss)
        train_accuracies.append(train_accuracy)
        validation_accuracies.append(validation_accuracy)

        dusk = t.time()
        print(f'elapsed time: {dusk - dawn:.2f} s')

        logger.save_accuracy_loss(epoch, train_accuracy, validation_accuracy, train_loss, validation_loss)

        if validation_accuracy > best_accuracy:
            best_accuracy = validation_accuracy
            misc.save_model(savepath, model, epoch, train_loss, validation_loss)
        elif epoch % 10 == 0:
            misc.save_model(savepath, model, epoch, train_loss, validation_loss)
        print(f'******** END EPOCH {epoch} ********\n')

    fig_accuracy, ax_accuracy = plt.subplots()
    fig_loss, ax_loss = plt.subplots()

    accuracy_graph = Graph(savepath, fig_accuracy, ax_accuracy, loss=False)
    loss_graph = Graph(savepath, fig_loss, ax_loss, loss=True)

    accuracy_graph.create(train_accuracies, validation_accuracies)
    loss_graph.create(train_losses, validation_losses)

    accuracy_graph.save()
    loss_graph.save()

    print('Fin.')


if __name__ == '__main__':
    main()
