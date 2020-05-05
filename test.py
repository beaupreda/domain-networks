"""
Computes the test accuracy of the model.

author: David-Alexandre Beaupre
date: 2020-05-02
"""

import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F

import utils.metrics as metrics
import utils.misc as misc
from datahandler.LITIV import LITIV
from datahandler.LITIVDataset import TestLITIVDataset
from models.concatnet import ConcatNet
from models.corrnet import CorrNet
from models.domainnet import DomainNet


def test(model: torch.nn.Module, loader: TestLITIVDataset, name: str, max_disp: int, bsize: int, n: int, cuda: bool) \
        -> float:
    """
    test function for either the proposed model, or its individual branches.
    :param model: torch model.
    :param loader: data loader.
    :param name: name of the model.
    :param max_disp: maximum disparity to match patches.
    :param bsize: batch size.
    :param n: disparity threshold.
    :param cuda: use GPU or not.
    :return: accuracy value.
    """
    print('testing...')
    model.eval()
    with torch.no_grad():
        correct = 0
        for i in range(0, loader.disparity.shape[0], bsize):
            print(f'\r{i + bsize} / {loader.disparity.shape[0]}', end='', flush=True)
            rgb, lwir, targets = loader.get_batch()

            disp = torch.arange(start=0, end=(max_disp + 1), dtype=torch.float32)
            disp = disp.repeat(repeats=(bsize, )).reshape(bsize, max_disp + 1)
            weight_corr = torch.zeros(size=(bsize, max_disp + 1), dtype=torch.float32)
            weight_concat = torch.zeros(size=(bsize, max_disp + 1), dtype=torch.float32)
            if cuda:
                rgb = rgb.cuda()
                lwir = lwir.cuda()
                targets = targets.cuda()
                disp = disp.cuda()
                weight_corr = weight_corr.cuda()
                weight_concat = weight_concat.cuda()

            frgb = model.rgb_features(rgb)
            flwir = model.lwir_features(lwir)

            for d in range(flwir.shape[3]):
                lw = flwir[:, :, :, d]
                lw = torch.unsqueeze(lw, dim=3)

                if name == 'corrnet':
                    correlation = torch.matmul(frgb, lw)
                    correlation = correlation.view(correlation.size(0), -1)
                    corr = torch.softmax(model.correlation_cls(correlation), dim=1)
                    weight_corr[:, d] = corr[:, 1]
                elif name == 'concatnet':
                    concatenation = torch.cat((F.relu(frgb), F.relu(lw)), dim=1)
                    concatenation = concatenation.view(concatenation.size(0), -1)
                    concat = torch.softmax(model.concat_cls(concatenation), dim=1)
                    weight_concat[:, d] = concat[:, 1]
                else:
                    correlation = torch.matmul(frgb, lw)
                    concatenation = torch.cat((F.relu(frgb), F.relu(lw)), dim=1)
                    correlation = correlation.view(correlation.size(0), -1)
                    concatenation = concatenation.view(concatenation.size(0), -1)
                    corr = torch.softmax(model.correlation_cls(correlation), dim=1)
                    concat = torch.softmax(model.concat_cls(concatenation), dim=1)
                    weight_corr[:, d] = corr[:, 1]
                    weight_concat[:, d] = concat[:, 1]

            if name == 'corrnet':
                w_corr = torch.softmax(weight_corr, dim=1)
                w_concat = torch.softmax(weight_corr, dim=1)
            elif name == 'concatnet':
                w_corr = torch.softmax(weight_concat, dim=1)
                w_concat = torch.softmax(weight_concat, dim=1)
            else:
                w_corr = torch.softmax(weight_corr, dim=1)
                w_concat = torch.softmax(weight_concat, dim=1)

            corr_d = torch.sum(w_corr * disp, dim=1)
            concat_d = torch.sum(w_concat * disp, dim=1)
            dp = (corr_d + concat_d) / 2.0
            correct += metrics.correct_matches_distance_n(dp, targets, n)

    accuracy = float(correct) / float(loader.disparity.shape[0])

    return accuracy


def main() -> None:
    torch.manual_seed(42)

    parser = argparse.ArgumentParser()
    parser.add_argument('--save', default='logs', help='folder to save logs from executions')
    parser.add_argument('--fold', type=int, default=1, help='which fold to test on (LITIV dataset only)')
    parser.add_argument('--model', default='concatnet', help='name of the model to train')
    parser.add_argument('--datapath', default='/home/beaupreda/litiv/datasets/litiv')
    parser.add_argument('--loadmodel', default='pretrained/concatnet/fold1.pt',
                        help='name of the model to load, if any')
    parser.add_argument('--max_disparity', type=int, default=64, help='maximum disparity in the dataset')
    parser.add_argument('--patch_size', type=int, default=18, help='half width of the left patch')
    parser.add_argument('--batch_size', type=int, default=100, help='test batch size')
    parser.add_argument('--n', type=int, default=3, help='threshold for the n pixel error function')
    parser.add_argument('--no_cuda', action='store_true', default=False, help='enables/disables GPU')
    args = parser.parse_args()

    dataset = LITIV(root=args.datapath, psize=args.patch_size, fold=args.fold,)
    dataloader = TestLITIVDataset(dataset.rgb['test'], dataset.lwir['test'], dataset.disp['test'], args)

    args.cuda = not args.no_cuda and torch.cuda.is_available()

    print(f'loading model...')
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

    accuracy = test(model, dataloader, args.model, args.max_disparity, args.batch_size, args.n, args.cuda)
    print(f'\ntest accuracy: {accuracy * 100:.2f}')

    print('Fin.')


if __name__ == '__main__':
    main()
