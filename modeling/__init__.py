# encoding: utf-8
"""
@author:  sherlock
@contact: sherlockliao01@gmail.com
"""

from .baseline import Baseline


def build_model(args):
    model = Baseline(args.num_classes, args.last_stride, args.pretrain_path, args.neck, args.neck_feat, args.base_model_name, args.pretrain_choice)
    return model
