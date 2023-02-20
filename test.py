# -*- coding: utf-8 -*-
# @Time    : 2022/3/21 15:07
# @Author  : hx
# @FileName: main.py
# @Software: PyCharm
from MBA_6branch import args
from trainer import Trainer
from utils.functions import checkpoint
from ptflops import get_model_complexity_info
from modeling import build_model
from dataset_loader import make_data_loader
import os
from retrieve import make_model
def main(args):
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_ids
    args.test_only = True
    #args.re_rank = True
    #args.neck_feat = 'before'
    # 生成dataloader

    train_loader, query_loader, gallery_loader, num_classes = make_data_loader(args)  # 由本工程代码加载含weights训练集
    args.num_classes = num_classes
    # 由本工程代码加载model
    from modeling.MBA import MBA_Net
    model = MBA_Net(args)

    model.load_param('/../experiment/{}/model/model_best.pt'.format(args.save))

    ckpt = checkpoint(args)
    # 生成losses
    loss = None

    # 生成训练器
    trainer = Trainer(args, model, loss, ckpt)
    trainer.test(query_loader, gallery_loader)

if __name__ == '__main__':

    main(args)
