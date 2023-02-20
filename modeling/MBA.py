# -*- coding: utf-8 -*-
# @Time    : 2022/6/6 20:21
# @Author  : hx
# @FileName: MBA.py
# @Software: PyCharm
from torchvision.models.resnet import resnet50, Bottleneck, resnet34, resnet101,BasicBlock,resnet152
from modeling.layers.non_local import Non_local
import torch
from torch import nn
import copy, os
from modeling.MGA import TransformerBlock_ViT, CrossTransformerBlock
from utils.gem_pool import GeneralizedMeanPoolingP
def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        nn.init.constant_(m.bias, 0.0)
    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)

class MBA_Net(nn.Module):
    "Multiple Branch Attention Net"
    def __init__(self, args):
        super(MBA_Net, self).__init__()
        num_classes = args.num_classes
        self.in_planes = 2048
        NL_21 = Non_local(512)
        NL_22 = Non_local(512)
        NL_31 = Non_local(1024)
        NL_32 = Non_local(1024)
        NL_33 = Non_local(1024)
        resnet = resnet50(pretrained=True)
        self.resnetNL = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2[:-1],
            NL_21,
            resnet.layer2[-1],
            NL_22,
            resnet.layer3[:-2],
            NL_31,
            resnet.layer3[-2],
            NL_32,
            resnet.layer3[-1],
            NL_33,
        )
        # last stride=1
        self.res_conv5_g = nn.Sequential(
            Bottleneck(1024, 512, downsample=nn.Sequential(nn.Conv2d(1024, 2048, 1,bias=False), nn.BatchNorm2d(2048))),
            Bottleneck(2048, 512),
            Bottleneck(2048, 512))
        self.res_conv5_g.load_state_dict(resnet.layer4.state_dict())

        self.res_conv5_p1 = resnet.layer4
        self.res_conv5_p2 = copy.deepcopy(self.res_conv5_p1)
        # self.backbone = nn.Sequential(self.resnetNL, res_conv5)  # (64,2048,8,4)

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        #self.pool_gem = GeneralizedMeanPoolingP()

        self.transformer = TransformerBlock_ViT(2048, 32,dropout=0.5)
        self.cross_transformer = CrossTransformerBlock(2048, 32,dropout=0.5)

        #reduction_g = nn.Sequential(nn.Conv2d(2048, 1024, 1, bias=False), nn.BatchNorm2d(1024), nn.ReLU())
        reduction = nn.Sequential(nn.Conv2d(2048, 512, 1, bias=False), nn.BatchNorm2d(512), nn.ReLU())
        reduction256 = nn.Sequential(nn.Conv2d(2048, 256, 1, bias=False), nn.BatchNorm2d(256), nn.ReLU())

        #self._init_reduction(reduction_g)
        self._init_reduction(reduction)
        self._init_reduction(reduction256)
        self.reduction_g1 = copy.deepcopy(reduction)
        self.reduction_g2 = copy.deepcopy(reduction)
        self.reduction_p11 = copy.deepcopy(reduction256)
        self.reduction_p12 = copy.deepcopy(reduction256)
        self.reduction_p21 = copy.deepcopy(reduction256)
        self.reduction_p22 = copy.deepcopy(reduction256)

        # self.fc_id_2048_0 = nn.Linear(2048, num_classes)
        self.fc_g1_512 = nn.Linear(512, num_classes)
        self.fc_g2_512 = nn.Linear(512, num_classes)
        self.fc_p11_256 = nn.Linear(256, num_classes)
        self.fc_p12_256 = nn.Linear(256, num_classes)
        self.fc_p21_256 = nn.Linear(256, num_classes)
        self.fc_p22_256 = nn.Linear(256, num_classes)

        self._init_fc(self.fc_g1_512)
        self._init_fc(self.fc_g2_512)
        self._init_fc(self.fc_p11_256)
        self._init_fc(self.fc_p12_256)
        self._init_fc(self.fc_p21_256)
        self._init_fc(self.fc_p22_256)
        self.neck_feat = args.neck_feat
        self.neck = args.neck
        if self.neck == 'bnneck':
            self.bottleneck_g1 = nn.BatchNorm1d(512)
            self.bottleneck_g1.bias.requires_grad_(False)  # no shift
            self.bottleneck_g1.apply(weights_init_kaiming)
            self.bottleneck_g2 = copy.deepcopy(self.bottleneck_g1)

            self.bottleneck_p11 = nn.BatchNorm1d(256)
            self.bottleneck_p11.bias.requires_grad_(False)  # no shift
            self.bottleneck_p11.apply(weights_init_kaiming)
            self.bottleneck_p12 = copy.deepcopy(self.bottleneck_p11)
            self.bottleneck_p21 = copy.deepcopy(self.bottleneck_p11)
            self.bottleneck_p22 = copy.deepcopy(self.bottleneck_p11)

    @staticmethod
    def _init_reduction(reduction):
        # conv
        nn.init.kaiming_normal_(reduction[0].weight, mode='fan_in')
        # nn.init.constant_(reduction[0].bias, 0.)

        # bn
        nn.init.normal_(reduction[1].weight, mean=1., std=0.02)
        nn.init.constant_(reduction[1].bias, 0.)
    @staticmethod
    def _init_fc(fc):
        nn.init.kaiming_normal_(fc.weight, mode='fan_out')
        # nn.init.normal_(fc.weight, std=0.001)
        nn.init.constant_(fc.bias, 0.)

    def forward(self, x):

        x = self.resnetNL(x) # [64, 1024, 16, 8]

        f_g = self.res_conv5_g(x) # 2048
        f_p1 = self.res_conv5_p1(x)
        f_p2 = self.res_conv5_p2(x)

        f_g1 = self.pool(f_g)
        f_g2 = self.max_pool(f_g)
        f_p11 = self.pool(self.transformer(f_p1))
        f_p12 = self.max_pool(self.transformer(f_p1))
        f_p21 = self.pool(self.cross_transformer(f_p2,f_p1))
        f_p22 = self.max_pool(self.cross_transformer(f_p2,f_p1))
        #print('f_p2', f_g.size(), f_p1.size(), f_p2.size())

        f_g1 = self.reduction_g1(f_g1).squeeze(dim=3).squeeze(dim=2)
        f_g2 = self.reduction_g2(f_g2).squeeze(dim=3).squeeze(dim=2)
        f_p11 = self.reduction_p11(f_p11).squeeze(dim=3).squeeze(dim=2)
        f_p12 = self.reduction_p12(f_p12).squeeze(dim=3).squeeze(dim=2)
        f_p21 = self.reduction_p21(f_p21).squeeze(dim=3).squeeze(dim=2)
        f_p22 = self.reduction_p22(f_p22).squeeze(dim=3).squeeze(dim=2)
        # print('feature.size()', feature.size())

        if  self.neck == 'bnneck':
            f_g1_norm = self.bottleneck_g1(f_g1)  # normalize for angular softmax
            f_g2_norm = self.bottleneck_g2(f_g2)  # normalize for angular softmax
            f_p11_norm = self.bottleneck_p11(f_p11)  # normalize for angular softmax
            f_p12_norm = self.bottleneck_p12(f_p12)  # normalize for angular softmax
            f_p21_norm = self.bottleneck_p21(f_p21)  # normalize for angular softmax
            f_p22_norm = self.bottleneck_p22(f_p22)  # normalize for angular softmax
        else:
            f_g1_norm = f_g1
            f_g2_norm = f_g2
            f_p11_norm = f_p11
            f_p12_norm = f_p12
            f_p21_norm = f_p21
            f_p22_norm = f_p22

        if self.training:
            score = self.fc_g1_512(f_g1_norm)
            score1 = self.fc_g2_512(f_g2_norm)
            score2 = self.fc_p11_256(f_p11_norm)
            score3 = self.fc_p12_256(f_p12_norm)
            score4 = self.fc_p21_256(f_p21_norm)
            score5 = self.fc_p22_256(f_p22_norm)
            return f_g1, f_g2, f_p11, f_p12,f_p21, f_p22, score, score1, score2, score3, score4, score5  # global feature for triplet loss
        else:
            if self.neck_feat == 'after':
                # print("Test with feature after BN")
                feature_final = torch.cat([f_g1_norm,f_g2_norm, f_p11_norm, f_p12_norm, f_p21_norm, f_p22_norm], dim=1)
                return feature_final
            else:
                # print("Test with feature before BN")
                feature_final = torch.cat([f_g1,f_g2, f_p11, f_p12,f_p21, f_p22], dim=1)
                return feature_final


    def save(self, apath, epoch, is_best=False):
        target = self
        torch.save(
            target.state_dict(),
            os.path.join(apath, 'model', 'model_{}.pt'.format(epoch))
        )
        if is_best:
            torch.save(
                target.state_dict(),
                os.path.join(apath, 'model', 'model_best.pt')
            )

    def load_param(self, trained_path):
        param_dict = torch.load(trained_path)
        for i in param_dict:
            self.state_dict()[i].copy_(param_dict[i])
