# -*- coding: utf-8 -*-
# @Time    : 2022/6/23 14:50
# @Author  : hx
# @FileName: MBA_6branch.py
# @Software: PyCharm

import numpy as np
from torch.backends import cudnn
import random
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from loss.triplet_loss import TripletLoss, CrossEntropyLabelSmooth


class Loss(nn.modules.loss._Loss):
    def __init__(self, args, ckpt):
        super(Loss, self).__init__()
        print('[INFO] Making loss...')

        self.nGPU = args.nGPU
        self.args = args
        self.loss = []
        self.loss_module = nn.ModuleList()
        for loss in args.loss.split('+'):
            weight, loss_type = loss.split('*')
            if loss_type == 'CrossEntropy':
                loss_function = nn.CrossEntropyLoss()
            elif loss_type == 'Triplet':
                loss_function = TripletLoss(args.margin)
            elif loss_type == 'CrossEntropyLabelSmooth':
                loss_function = CrossEntropyLabelSmooth(self.args.num_classes)
            else:
                raise RuntimeError

            self.loss.append({
                'type': loss_type,
                'weight': float(weight),
                'function': loss_function
            })

        if len(self.loss) > 1:
            self.loss.append({'type': 'Total', 'weight': 0, 'function': None})

        for l in self.loss:
            if l['function'] is not None:
                print('{:.3f} * {}'.format(l['weight'], l['type']))
                self.loss_module.append(l['function'])

        self.log = torch.Tensor()

        device = torch.device('cpu' if args.cpu else 'cuda')
        self.loss_module.to(device)

        if args.load != '': self.load(ckpt.dir, cpu=args.cpu)
        if not args.cpu and args.nGPU > 1:
            self.loss_module = nn.DataParallel(
                self.loss_module, range(args.nGPU)
            )

    def forward(self, outputs, labels):
        losses = []
        for i, l in enumerate(self.loss):
            if l['type'] == 'Triplet':
                loss = [l['function'](output, labels) for output in outputs[:6]]
                loss = sum(loss) / len(loss)
                effective_loss = l['weight'] * loss
                losses.append(effective_loss)
                self.log[-1, i] += effective_loss.item()
            elif l['function'] is not None:
                loss = [l['function'](output, labels) for output in outputs[6:]]
                loss = sum(loss) / len(loss)
                effective_loss = l['weight'] * loss
                losses.append(effective_loss)
                self.log[-1, i] += effective_loss.item()
            else:
                pass
        loss_sum = sum(losses)
        if len(self.loss) > 1:
            self.log[-1, -1] += loss_sum.item()

        return loss_sum

    def start_log(self):
        self.log = torch.cat((self.log, torch.zeros(1, len(self.loss))))

    def end_log(self, batches):
        self.log[-1].div_(batches)

    def display_loss(self, batch):
        n_samples = batch + 1
        log = []
        for l, c in zip(self.loss, self.log[-1]):
            log.append('[{}: {:.4f}]'.format(l['type'], c / n_samples))

        return ''.join(log)

    def plot_loss(self, apath, epoch):
        axis = np.linspace(1, epoch, epoch)
        for i, l in enumerate(self.loss):
            label = '{} Loss'.format(l['type'])
            fig = plt.figure()
            plt.title(label)
            plt.plot(axis, self.log[:, i].numpy(), label=label)
            plt.legend()
            plt.xlabel('Epochs')
            plt.ylabel('Loss')
            plt.grid(True)
            plt.savefig('{}/loss_{}.jpg'.format(apath, l['type']))
            plt.close(fig)

    def step(self):
        for l in self.get_loss_module():
            if hasattr(l, 'scheduler'):
                l.scheduler.step()

    def get_loss_module(self):
        if self.nGPU == 1:
            return self.loss_module
        else:
            return self.loss_module.module

    def save(self, apath):
        torch.save(self.state_dict(), os.path.join(apath, 'loss.pt'))
        torch.save(self.log, os.path.join(apath, 'loss_log.pt'))

    def load(self, apath, cpu=False):
        if cpu:
            kwargs = {'map_location': lambda storage, loc: storage}
        else:
            kwargs = {}

        self.load_state_dict(torch.load(
            os.path.join(apath, 'loss.pt'),
            **kwargs
        ))
        self.log = torch.load(os.path.join(apath, 'loss_log.pt'))
        for l in self.loss_module:
            if hasattr(l, 'scheduler'):
                for _ in range(len(self.log)): l.scheduler.step()


from trainer import Trainer
from utils.functions import checkpoint

from dataset_loader import make_data_loader
import os
import argparse
# Options
# --------
parser = argparse.ArgumentParser(description='baseline net')
# GPU
parser.add_argument('--nThread', type=int, default=4, help='number of threads for data loading/num_workers')
parser.add_argument('--cpu', action='store_true', help='use cpu only')
parser.add_argument('--nGPU', type=int, default=4, help='number of GPUs')
parser.add_argument('--gpu_ids', default='2,3,0,1', type=str, help='gpu_ids: e.g. 0  0,1,2  0,2')
# data
parser.add_argument("--datadir", type=str, default="/../data", help='dataset directory')
parser.add_argument('--dataset_name', type=str, default='occluded_dukemtmc', help='train dataset name') #msmt17,
# data_load
parser.add_argument('--height', type=int, default=384, help='height of the input image')
parser.add_argument('--width', type=int, default=128, help='width of the input image')
parser.add_argument("--batchid", type=int, default=16, help='the ids for batch')
parser.add_argument("--batchimage", type=int, default=4, help='the images of per id')
parser.add_argument("--batchtest", type=int, default=32, help='input batch size for test')
parser.add_argument("--random_erasing_probability", type=float, default=0.5, help='')
# training
parser.add_argument('--test_only', action='store_true', help='set this option to test the model')
parser.add_argument('--reset', default=False, help='reset the training')
parser.add_argument('--seed', default=7, help='random seed for ensure result')
parser.add_argument("--epochs", type=int, default=200, help='number of epochs to train')
parser.add_argument('--test_every', type=int, default=50, help='do test per every N epochs')
parser.add_argument("--margin", type=float, default=0.3, help='')
parser.add_argument("--re_rank", default=False, help='')
parser.add_argument("--last_stride", default=1, help='1 or 2')
parser.add_argument("--neck", default='bnneck', help="If train with BNNeck, 'bnneck' or 'no'")
parser.add_argument("--neck_feat", default='after', help="Which feature of BNNeck to be used for test, before or after BNNneck, options: 'before' or 'after'")
# loss
parser.add_argument('--loss', type=str, default='1.5*CrossEntropyLabelSmooth+2.0*Triplet', help='loss function configuration')  # ArcFaceLoss,CrossEntropyLabelSmooth
# model and save
parser.add_argument("--base_model_name", type=str, default='resnetNL', help='Name of backbone')
parser.add_argument("--pretrain_choice", type=str, default='imagenet', help='imagenet or self')
parser.add_argument("--resume", type=int, default=0, help='resume from specific checkpoint')
parser.add_argument('--save', type=str, default='', help='file name to save')  #
parser.add_argument('--load', type=str, default='', help='file name to load')
# optimizer and lr_scheduler
parser.add_argument("--lr", type=float, default=3.5e-3, help='learning rate')
parser.add_argument('--optimizer', default='SGD', choices=('SGD', 'ADAM', 'NADAM', 'RMSprop'), help='optimizer to use (SGD | ADAM | NADAM | RMSprop)')
parser.add_argument('--momentum', type=float, default=0.9, help='SGD momentum')
parser.add_argument('--dampening', type=float, default=0, help='SGD dampening')
parser.add_argument('--nesterov', action='store_true', help='SGD nesterov')
parser.add_argument('--beta1', type=float, default=0.9, help='ADAM beta1')
parser.add_argument('--beta2', type=float, default=0.999, help='ADAM beta2')
parser.add_argument('--amsgrad', default=True, help='ADAM amsgrad')
parser.add_argument('--epsilon', type=float, default=1e-8, help='ADAM epsilon for numerical stability')
parser.add_argument('--gamma', type=float, default=0.1, help='learning rate decay factor for step decay')
parser.add_argument('--weight_decay', type=float, default=5e-4, help='weight decay')
parser.add_argument('--lr_decay', type=int, default=60, help='learning rate decay per N epochs')
parser.add_argument('--decay_type', type=str, default='warmup_cosine', help='learning rate decay type')
parser.add_argument('--warmup_iters', type=int, default=10, help='warmup in first N epochs')
args = parser.parse_args()

def main(args):

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_ids
    # 模仿transreid
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    cudnn.benchmark = False
    cudnn.deterministic = True
    # 生成dataloader
    train_loader, query_loader, gallery_loader, num_classes = make_data_loader(args)
    parser.add_argument('--num_classes', type=int, default=num_classes, help='num_classes of train set')
    args = parser.parse_args()
    # 由本工程代码加载model
    from modeling.MBA import MBA_Net
    model = MBA_Net(args)
    # save and log things
    ckpt = checkpoint(args)
    # 生成losses
    loss = Loss(args, ckpt)
    # 生成训练器
    trainer = Trainer(args, model, loss, ckpt)

    for epoch in range(args.epochs):
        trainer.train(train_loader)
        if (epoch+1) % args.test_every == 0:
            trainer.test(query_loader, gallery_loader)


if __name__ == '__main__':
    main(args)


