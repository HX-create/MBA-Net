# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""
from bisect import bisect_right
import torch
import torch.optim.lr_scheduler as lrs
import torch.optim as optim
# FIXME ideally this would be achieved with a CombinedLRScheduler,
# separating MultiStepLR with WarmupLR
# but the current LRScheduler design doesn't allow it

class WarmupMultiStepLR(torch.optim.lr_scheduler._LRScheduler):
    def __init__(
        self,
        optimizer,
        milestones,  # step epoch
        gamma=0.1,
        warmup_init_lr_rate = 0.01,  # 初始学习率为稳定学习率的比率
        warmup_iters=10,
        warmup_method="linear",
        last_epoch=-1,
    ):
        if not list(milestones) == sorted(milestones):
            raise ValueError(
                "Milestones should be a list of" " increasing integers. Got {}",
                milestones,
            )

        if warmup_method not in ("constant", "linear"):
            raise ValueError(
                "Only 'constant' or 'linear' warmup_method accepted"
                "got {}".format(warmup_method)
            )
        self.milestones = milestones
        self.gamma = gamma
        self.warmup_init_lr_rate = warmup_init_lr_rate
        self.warmup_iters = warmup_iters
        self.warmup_method = warmup_method
        super(WarmupMultiStepLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        lr = [base_lr for base_lr in self.base_lrs][0]  # 就是optimizer[lr],改list为数，该值一直在更新
        init_lr = lr * self.warmup_init_lr_rate  # 初始学习率
        if self.last_epoch < self.warmup_iters:
            if self.warmup_method == "linear":
                alpha = self.last_epoch / self.warmup_iters
                lr = init_lr + (lr - init_lr) * alpha
                # print('self.last_epoch', self.last_epoch)
        return [lr * self.gamma ** bisect_right(self.milestones, self.last_epoch)]
from math import cos, pi
class WarmupAndCosine(torch.optim.lr_scheduler._LRScheduler):
    def __init__(
            self,
            optimizer,
            warmup_init_lr_rate=0.01,  # 初始学习率为稳定学习率的比率
            warmup_iters=10,
            warmup_method="linear",
            max_epoch=120,
            last_epoch=-1,
            lr_min=7e-7,  # fastreid reference
    ):
        if warmup_method not in ("constant", "linear"):
            raise ValueError(
                "Only 'constant' or 'linear' warmup_method accepted"
                "got {}".format(warmup_method)
            )
        self.warmup_init_lr_rate = warmup_init_lr_rate
        self.warmup_iters = warmup_iters
        self.warmup_method = warmup_method
        self.maxlr_epoch = max_epoch / 2
        self.lr_min = lr_min
        self.max_epoch = max_epoch
        super(WarmupAndCosine, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        lr = [base_lr for base_lr in self.base_lrs][0]  # 就是optimizer[lr],改list为数，该值一直在更新
        init_lr = lr * self.warmup_init_lr_rate  # 初始学习率
        if self.last_epoch < self.warmup_iters:
            alpha = self.last_epoch / self.warmup_iters
            lr = init_lr + (lr - init_lr) * alpha

        if self.last_epoch > self.maxlr_epoch:
            lr = self.lr_min + (lr - self.lr_min) * (
                    1 + cos(pi * (self.last_epoch - self.maxlr_epoch) / (self.max_epoch - self.maxlr_epoch))) / 2

        return [lr]


def make_optimizer(args, model):
    trainable = filter(lambda x: x.requires_grad, model.parameters())

    if args.optimizer == 'SGD':
        optimizer_function = optim.SGD
        kwargs = {
            'momentum': args.momentum,
            'dampening': args.dampening,
            'nesterov': args.nesterov
        }
    elif args.optimizer == 'ADAM':
        optimizer_function = optim.Adam
        kwargs = {
            'betas': (args.beta1, args.beta2),
            'eps': args.epsilon,
            'amsgrad': args.amsgrad
        }
    elif args.optimizer == 'RMSprop':
        optimizer_function = optim.RMSprop
        kwargs = {
            'eps': args.epsilon,
            'momentum': args.momentum
        }
    else:
        raise Exception()

    kwargs['lr'] = args.lr
    kwargs['weight_decay'] = args.weight_decay

    return optimizer_function(trainable, **kwargs)


def make_scheduler(args, optimizer):
    if args.decay_type == 'step':
        scheduler = lrs.StepLR(
            optimizer,
            step_size=args.lr_decay,
            gamma=args.gamma
        )
    elif args.decay_type.find('step') >= 0:
        milestones = args.decay_type.split('_')
        milestones.pop(0)
        milestones = list(map(lambda x: int(x), milestones))
        scheduler = lrs.MultiStepLR(
            optimizer,
            milestones=milestones,
            gamma=args.gamma
        )
    elif args.decay_type.find('warmup') >= 0:
        if args.decay_type.find('cosine') >= 0:
            scheduler = WarmupAndCosine(
                optimizer=optimizer,
                warmup_iters=args.warmup_iters,
                max_epoch=args.epochs,
            )
        else:
            milestones = args.decay_type.split('_')
            milestones.pop(0)
            milestones = list(map(lambda x: int(x), milestones))
            scheduler = WarmupMultiStepLR(
                optimizer,
                warmup_iters=args.warmup_iters,
                milestones=milestones,
            )
    else:
        raise RuntimeError

    scheduler.last_epoch = -1
    return scheduler