# -*- coding: utf-8 -*-
# @Time    : 2022/3/21 14:49
# @Author  : hx
# @FileName: trainer.py
# @Software: PyCharm
import os
import torch
import numpy as np
from utils.lr_scheduler import make_optimizer, make_scheduler
from scipy.spatial.distance import cdist
from utils.functions import cmc, mean_ap
from utils.re_ranking import re_ranking
import time
class Trainer:
    def __init__(self, args, model, loss, ckpt):
        self.args = args
        if args.height == 384:
            self.feature_length = model.in_planes
        else:  # ViT Baseline
            self.feature_length = 768
        self.ckpt = ckpt
        self.model = model
        self.optimizer = make_optimizer(args, self.model)
        self.scheduler = make_scheduler(args, self.optimizer)
        self.device = torch.device('cpu' if args.cpu else 'cuda')
        self.model.to(self.device)
        self.loss = loss
        self.lr = 0.


        if args.load != '':
            self.optimizer.load_state_dict(
                torch.load(os.path.join(ckpt.dir, 'optimizer.pt'))
            )
            for _ in range(len(ckpt.log) * args.test_every): self.scheduler.step()
            self.model.load_param('/LabData/hx/ReID_Baseline_New/experiment/{}/model/model_{}.pt'.format(args.save, 'best'))

        if not self.args.cpu and self.args.nGPU > 1:
            self.model = torch.nn.DataParallel(self.model, range(args.nGPU))

    def train(self, train_loader):
        self.scheduler.step()
        self.loss.step()
        epoch = self.scheduler.last_epoch + 1
        lr = self.scheduler.get_last_lr()[0]
        if lr != self.lr:
            self.ckpt.write_log('[INFO] Epoch: {}\tLearning rate: {:.2e}'.format(epoch, lr))
            self.lr = lr
        self.loss.start_log()  # 记录loss的tensor
        self.model.train()

        for batch, (inputs, labels) in enumerate(train_loader):
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.loss(outputs, labels)
            loss.backward()
            self.optimizer.step()

            self.ckpt.write_log('\r[INFO] [{}/{}]\t{}/{}\t{}'.format(
                epoch, self.args.epochs,
                batch + 1, len(train_loader),
                self.loss.display_loss(batch)),
            end='' if batch+1 != len(train_loader) else '\n')
            # if batch == 1:
            #     break


        self.loss.end_log(len(train_loader))


    def test(self, query_loader, gallery_loader):
        epoch = self.scheduler.last_epoch + 1
        self.ckpt.write_log('\n[INFO] Test:')
        self.model.eval()

        self.ckpt.add_log(torch.zeros(1, 5))
        start = time.time()
        print("start", start)
        with torch.no_grad():
            #t1 = time.time()
            gf, gp, gc = self.extract_feature(gallery_loader)
            qf, qp, qc = self.extract_feature(query_loader)  # feature, pids, camid
            #print('extract_feature use {}s'.format(time.time()-t1))

        if self.args.re_rank:
            print('start re-ranking!!!')
            q_g_dist = np.dot(qf, np.transpose(gf))
            q_q_dist = np.dot(qf, np.transpose(qf))
            g_g_dist = np.dot(gf, np.transpose(gf))
            dist = re_ranking(q_g_dist, q_q_dist, g_g_dist)
        else:
            dist = cdist(qf, gf)
        print('start evaluating!!!')
        r = cmc(dist, qp, gp, qc, gc,
                separate_camera_set=False,
                single_gallery_shot=False,
                first_match_break=True)
        end = time.time()
        print("using time: ", end-start)
        m_ap = mean_ap(dist, qp, gp, qc, gc)

        self.ckpt.log[-1, 0] = m_ap
        self.ckpt.log[-1, 1] = r[0]
        self.ckpt.log[-1, 2] = r[2]
        self.ckpt.log[-1, 3] = r[4]
        self.ckpt.log[-1, 4] = r[9]
        best = self.ckpt.log.max(0)
        self.ckpt.write_log(
            '[INFO] mAP: {:.4f} rank1: {:.4f} rank3: {:.4f} rank5: {:.4f} rank10: {:.4f} (Best: {:.4f} @epoch {})'.format(
            m_ap,
            r[0], r[2], r[4], r[9],
            best[0][0],
            (best[1][0] + 1)*self.args.test_every
            )
        )
        if not self.args.test_only:
            self.ckpt.save(self, epoch, is_best=((best[1][0] + 1)*self.args.test_every == epoch))

    def fliphor(self, inputs):
        inv_idx = torch.arange(inputs.size(3)-1,-1,-1).long()  # N x C x H x W
        return inputs.index_select(3,inv_idx)

    def extract_feature(self, loader):
        features = torch.FloatTensor()
        pids = []
        camids = []

        for (inputs, pid, camid) in loader:
            pids.extend(np.asarray(pid))
            camids.extend(np.asarray(camid))
            ff = torch.FloatTensor(inputs.size(0), self.feature_length).zero_()
            # 含翻转增强，将两个output相加。
            for i in range(2):
                if i == 1:
                    inputs = self.fliphor(inputs)

                input_img = inputs.to(self.device)
                if not self.args.cpu and self.args.nGPU > 1:
                    outputs = self.model.module(input_img) # eval(),=feature  # module多卡并行时卡死解决
                else:
                    outputs = self.model(input_img)
                f = outputs.data.cpu()  # 不计算grad
                ff = ff + f
            ###
            # input_img = inputs.to(self.device)
            # if not self.args.cpu and self.args.nGPU > 1:
            #     outputs = self.model.module(input_img)  # eval(),=feature  # module多卡并行时卡死解决
            # else:
            #     outputs = self.model(input_img)
            # ff = outputs.data.cpu()
            ###
            feats = torch.nn.functional.normalize(ff, dim=1, p=2)

            features = torch.cat((features, feats), 0)
            #print('feat',features.size())
        return features.numpy(), pids, camids

    def terminate(self):
        epoch = self.scheduler.last_epoch + 1
        return epoch >= self.args.epochs



