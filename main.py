#!/usr/bin/env python
# -*- coding: utf-8 -*-


from __future__ import print_function
import os
import gc
import argparse
import numpy as np
from tensorboardX import SummaryWriter
from tqdm import tqdm
import time

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader



from model import DeepUME
from util import timer, transform_point_cloud, chamfer, RRMSE
from data import Data


# Part of the code is referred from: https://github.com/WangYueFt/dcp

class IOStream:
    def __init__(self, path):
        self.f = open(path, 'a')

    def cprint(self, text):
        print(text)
        self.f.write(text + '\n')
        self.f.flush()

    def close(self):
        self.f.close()


def _init_(args):
    if not os.path.exists('checkpoints'):
        os.makedirs('checkpoints')
    if not os.path.exists('checkpoints/' + args.exp_name):
        os.makedirs('checkpoints/' + args.exp_name)
    if not os.path.exists('checkpoints/' + args.exp_name + '/' + 'models'):
        os.makedirs('checkpoints/' + args.exp_name + '/' + 'models')
    os.system('cp main.py checkpoints' + '/' + args.exp_name + '/' + 'main.py.backup')
    os.system('cp model.py checkpoints' + '/' + args.exp_name + '/' + 'model.py.backup')
    os.system('cp data.py checkpoints' + '/' + args.exp_name + '/' + 'data.py.backup')


def test_one_epoch(args, net, test_loader, given_q1=None, given_q2=None):
    net.eval()

    total_loss = 0
    total_maxmin = 0
    total_R_RMSE = 0
    total_t_RMSE = 0
    num_examples = 0
    for item, (src, target, rotation_gt, translation_gt) in enumerate(tqdm(test_loader)):

        if args.noise == 'bernoulli':
            if given_q1 is not None and given_q2 is not None:
                q1, q2 = given_q1, given_q2
            else:
                q1, q2 = np.random.uniform(low=0.2, high=1, size=2)
            indexes_1 = [bool(np.random.binomial(n=1, p=q1)) for _ in range(1024)]
            indexes_2 = [bool(np.random.binomial(n=1, p=q2)) for _ in range(1024)]

            while sum(indexes_1) < 200:  # stay with at least 200 points
                indexes_1 = [bool(np.random.binomial(n=1, p=q1)) for _ in range(1024)]
            while sum(indexes_2) < 200:  # stay with at least 200 points
                indexes_2 = [bool(np.random.binomial(n=1, p=q2)) for _ in range(1024)]
            src = src[:, :, indexes_1]
            target = target[:, :, indexes_2]

        src = src.cuda()
        target = target.cuda()

        rotation_gt = rotation_gt.cuda()
        translation_gt = translation_gt.cuda()

        batch_size = src.size(0)
        num_examples += batch_size
        rotation_pred, translation_pred = net(src, target)

        loss, maxmin = chamfer(target, transform_point_cloud(src, rotation_pred, translation_pred))
        R_RMSE = RRMSE(rotation_pred, rotation_gt)
        t_RMSE = torch.sqrt(torch.mean((translation_pred - translation_gt) ** 2))

        total_loss += loss.item() * batch_size
        total_maxmin += maxmin.item() * batch_size
        total_R_RMSE += R_RMSE * batch_size
        total_t_RMSE += t_RMSE.item() * batch_size

    return total_loss * 1.0 / num_examples, \
           total_maxmin * 1.0 / num_examples, \
           total_R_RMSE * 1.0 / num_examples, \
           total_t_RMSE * 1.0 / num_examples


def test(args, net, test_loader, textio=None, given_q1=None, given_q2=None):
    chamfer, maxmin, R_RMSE, t_RMSE = test_one_epoch(args, net, test_loader, given_q1, given_q2)

    if textio:
        textio.cprint('==FINAL TEST==')
        textio.cprint('EPOCH:: %d, chamfer: %f, maxmin: %f, R_RMSE: %f, t_RMSE: %f'
                      % (-1, chamfer, maxmin, R_RMSE, t_RMSE))
    else:
        print('==FINAL TEST==')
        print('EPOCH:: %d, chamfer: %f, maxmin: %f, R_RMSE: %f, t_RMSE: %f'
              % (-1, chamfer, maxmin, R_RMSE, t_RMSE))
    return R_RMSE


def train_one_epoch(net, train_loader, opt):
    net.train()

    total_loss = 0
    total_maxmin = 0
    total_R_RMSE = 0
    total_t_RMSE = 0
    num_examples = 0
    for src, target, rotation_gt, translation_gt in tqdm(train_loader):
        src = src.cuda()
        target = target.cuda()
        rotation_gt = rotation_gt.cuda()
        translation_gt = translation_gt.cuda()

        batch_size = src.size(0)
        num_examples += batch_size
        rotation_pred, translation_pred = net(src, target)

        loss, maxmin = chamfer(target, transform_point_cloud(src, rotation_pred, translation_pred))
        R_RMSE = RRMSE(rotation_pred, rotation_gt)
        t_RMSE = torch.sqrt(torch.mean((translation_pred - translation_gt) ** 2))

        opt.zero_grad()
        loss.backward()
        opt.step()

        total_loss += loss.item() * batch_size
        total_maxmin += maxmin.item() * batch_size
        total_R_RMSE += R_RMSE * batch_size
        total_t_RMSE += t_RMSE.item() * batch_size

    return total_loss * 1.0 / num_examples, \
           total_maxmin * 1.0 / num_examples, \
           total_R_RMSE * 1.0 / num_examples, \
           total_t_RMSE * 1.0 / num_examples


def train(args, net, train_loader, test_loader, boardio, textio, checkpoint):
    opt = optim.Adam(net.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = MultiStepLR(opt, milestones=[75, 150, 200], gamma=0.1)
    best_test_loss = np.inf

    if checkpoint is not None:
        best_test_loss = checkpoint['min_loss']
        opt.load_state_dict(checkpoint['opt'])
        print(f'Continue training from epoch {args.start_epoch} and min loss {best_test_loss}')

    for epoch in range(args.start_epoch, args.epochs):
        train_total_loss, train_total_maxmin, train_total_R_RMSE, train_total_t_RMSE = train_one_epoch(net, train_loader, opt)
        test_total_loss, test_total_maxmin, test_total_R_RMSE, test_total_t_RMSE = test_one_epoch(args, net, test_loader)

        snap = {'epoch': epoch + 1,
                'net': net.state_dict(),
                'min_loss': best_test_loss,
                'opt': opt.state_dict(), }

        if best_test_loss >= test_total_loss:
            best_test_loss = test_total_loss
            snap['min_loss'] = best_test_loss
            torch.save(net.state_dict(), 'checkpoints/%s/models/model.best.t7' % args.exp_name)

        torch.save(snap, 'checkpoints/%s/models/model_snap.t7' % args.exp_name)
        torch.save(net.state_dict(), 'checkpoints/%s/models/model.t7' % args.exp_name)

        textio.cprint('==TRAIN==')
        textio.cprint('EPOCH:: %d, Loss: %f, Maxmin: %f, R_RMSE: %f, t_RMSE: %f'
                      % (epoch, train_total_loss, train_total_maxmin, train_total_R_RMSE, train_total_t_RMSE))

        textio.cprint('==TEST==')
        textio.cprint('EPOCH:: %d, Loss: %f, Maxmin: %f, R_RMSE: %f, t_RMSE: %f'
                      % (epoch, test_total_loss, test_total_maxmin, test_total_R_RMSE, test_total_t_RMSE))

        textio.cprint('==BEST TEST==')
        textio.cprint('EPOCH:: %d, Loss: %f'
                      % (epoch, best_test_loss))

        boardio.add_scalar('train/total_loss', train_total_loss, epoch)
        boardio.add_scalar('train/Maxmin', train_total_maxmin, epoch)
        boardio.add_scalar('train/R_RMSE', train_total_R_RMSE, epoch)
        boardio.add_scalar('train/t_RMSE', train_total_t_RMSE, epoch)

        boardio.add_scalar('test/total_loss', test_total_loss, epoch)
        boardio.add_scalar('test/Maxmin', test_total_maxmin, epoch)
        boardio.add_scalar('test/R_RMSE', test_total_R_RMSE, epoch)
        boardio.add_scalar('test/t_RMSE', test_total_t_RMSE, epoch)

        gc.collect()
        scheduler.step()


def options():
    parser = argparse.ArgumentParser(description='Point Cloud Registration')
    parser.add_argument('--exp_name', type=str, default='exp', metavar='N',
                        help='Name of the experiment')
    parser.add_argument('--eval', action='store_true', default=False,
                        help='Evaluate the model')

    # setting for data
    parser.add_argument('--test_dataset', type=str, default='FAUST', metavar='N',
                        choices=['ModelNet40', 'FAUST', 'Stanford'],
                        help='Dataset to use for testing')
    parser.add_argument('--num_points', type=int, default=1024, metavar='N',
                        help='Num of points to use')
    parser.add_argument('--batch_size', type=int, default=32, metavar='batch_size',
                        help='Size of batch)')
    parser.add_argument('--test_batch_size', type=int, default=10, metavar='batch_size',
                        help='Size of batch)')
    parser.add_argument('--sigma', default=None, type=float, metavar='N',
                        help='Simga of the Gaussian noise to use')
    parser.add_argument('--noise', type=str, default='zero_intersec', metavar='N',
                        choices=['sampling', 'zero_intersec', 'bernoulli', 'gaussian', ''],
                        help='Noise to use')

    # setting for transformer
    parser.add_argument('--emb_dims', type=int, default=3, metavar='N',
                        help='Dimension of embeddings')
    parser.add_argument('--n_blocks', type=int, default=1, metavar='N',
                        help='Num of blocks of encoder&decoder')
    parser.add_argument('--n_heads', type=int, default=1, metavar='N',
                        help='Num of heads in multiheadedattention')
    parser.add_argument('--ff_dims', type=int, default=64, metavar='N',
                        help='Num of dimensions of fc in transformer')
    parser.add_argument('--dropout', type=float, default=0.0, metavar='N',
                        help='Dropout ratio in transformer')

    # setting for on training
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='Learning rate, default: 0.001')
    parser.add_argument('--seed', type=int, default=1234, metavar='S',
                        help='Random seed')
    parser.add_argument('--epochs', type=int, default=250, metavar='N',
                        help='number of episode to train ')
    parser.add_argument('--start_epoch', type=int, default=0, metavar='N',
                        help='Manual epoch number (useful on restarts)')
    parser.add_argument('--resume', type=str, default=False, metavar='PATH',
                        help='Path to latest checkpoint')
    parser.add_argument('--pretrained', type=str, default='pretrained/deepume.t7', metavar='PATH',
                        help='Path to pretrained model file')

    args = parser.parse_args()
    return args


def main(args):
    start = time.time()

    torch.backends.cudnn.deterministic = True
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)

    boardio = SummaryWriter(log_dir='checkpoints/' + args.exp_name)
    _init_(args)

    textio = IOStream('checkpoints/' + args.exp_name + '/run.log')
    textio.cprint(str(args))


    net = DeepUME(args).cuda()
    if args.eval:
        args.test_batch_size = 1
        if args.pretrained == '':
            pretrained = 'checkpoints' + '/' + args.exp_name + '/models/model.best.t7'
        else:
            pretrained = args.pretrained
            print(pretrained)
        if not os.path.exists(pretrained):
            print("can't find pretrained model")
            return
        net.load_state_dict(torch.load(pretrained), strict=False)
        print("model has been loaded")

    checkpoint = None
    if args.resume:
        path = 'checkpoints' + '/' + args.exp_name + '/models/model_snap.t7'
        assert os.path.isfile(path)
        checkpoint = torch.load(path)
        net.load_state_dict(checkpoint['net'])
        args.start_epoch = checkpoint['epoch']

    trainset = Data(type='ModelNet40', partition='train', noise='sampling')
    testset = Data(type=args.test_dataset, partition='test', noise=args.noise)
    train_loader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(testset, batch_size=args.test_batch_size, shuffle=False, drop_last=False)

    if args.eval:
        test(args, net, test_loader, textio)
    else:
        train(args, net, train_loader, test_loader, boardio, textio, checkpoint)

    textio.cprint('==RUNNING TIME==')
    textio.cprint("%d:%d:%d" % timer(start, time.time()))
    print('FINISH')
    boardio.close()


if __name__ == '__main__':
    args_main = options()
    main(args_main)
