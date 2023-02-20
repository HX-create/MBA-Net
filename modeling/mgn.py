# -*- coding: utf-8 -*-
# @Time    : 2022/5/17 11:14
# @Author  : hx
# @FileName: mgn.py
# @Software: PyCharm
import copy
import os
import torch
from torch import nn
#from .backbones.resnet_ibn_a import resnet50_ibn_a
from torchvision.models.resnet import resnet50, Bottleneck
from .MGA import CrossTransformerBlock, TransformerBlock
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
class MGN(nn.Module):
    def __init__(self, args):
        super(MGN, self).__init__()
        num_classes = args.num_classes
        self.in_planes = 2048

        resnet = resnet50(pretrained=True)

        self.backone = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2,
            resnet.layer3[0],
        )

        res_conv4 = nn.Sequential(*resnet.layer3[1:])

        res_g_conv5 = resnet.layer4

        res_p_conv5 = nn.Sequential(
            Bottleneck(1024, 512, downsample=nn.Sequential(nn.Conv2d(1024, 2048, 1, bias=False), nn.BatchNorm2d(2048))),
            Bottleneck(2048, 512),
            Bottleneck(2048, 512))
        res_p_conv5.load_state_dict(resnet.layer4.state_dict())

        self.p1 = nn.Sequential(copy.deepcopy(res_conv4), copy.deepcopy(res_g_conv5))
        self.p2 = nn.Sequential(copy.deepcopy(res_conv4), copy.deepcopy(res_p_conv5))
        self.p3 = nn.Sequential(copy.deepcopy(res_conv4), copy.deepcopy(res_p_conv5))

        if args.pool == 'max':
            pool2d = nn.MaxPool2d
        elif args.pool == 'avg':
            pool2d = nn.AvgPool2d
        else:
            raise Exception()

        self.maxpool_zg_p1 = pool2d(kernel_size=(12, 4))
        self.maxpool_zg_p2 = pool2d(kernel_size=(24, 8))
        self.maxpool_zg_p3 = pool2d(kernel_size=(24, 8))
        self.maxpool_zp2 = pool2d(kernel_size=(12, 8))
        self.maxpool_zp3 = pool2d(kernel_size=(8, 8))

        reduction = nn.Sequential(nn.Conv2d(2048, args.feats, 1, bias=False), nn.BatchNorm2d(args.feats), nn.ReLU())

        self._init_reduction(reduction)
        self.reduction_0 = copy.deepcopy(reduction)
        self.reduction_1 = copy.deepcopy(reduction)
        self.reduction_2 = copy.deepcopy(reduction)
        self.reduction_3 = copy.deepcopy(reduction)
        self.reduction_4 = copy.deepcopy(reduction)
        self.reduction_5 = copy.deepcopy(reduction)
        self.reduction_6 = copy.deepcopy(reduction)
        self.reduction_7 = copy.deepcopy(reduction)

        # self.fc_id_2048_0 = nn.Linear(2048, num_classes)
        self.fc_id_2048_0 = nn.Linear(args.feats, num_classes)
        self.fc_id_2048_1 = nn.Linear(args.feats, num_classes)
        self.fc_id_2048_2 = nn.Linear(args.feats, num_classes)

        self.fc_id_256_1_0 = nn.Linear(args.feats, num_classes)
        self.fc_id_256_1_1 = nn.Linear(args.feats, num_classes)
        self.fc_id_256_2_0 = nn.Linear(args.feats, num_classes)
        self.fc_id_256_2_1 = nn.Linear(args.feats, num_classes)
        self.fc_id_256_2_2 = nn.Linear(args.feats, num_classes)

        self._init_fc(self.fc_id_2048_0)
        self._init_fc(self.fc_id_2048_1)
        self._init_fc(self.fc_id_2048_2)

        self._init_fc(self.fc_id_256_1_0)
        self._init_fc(self.fc_id_256_1_1)
        self._init_fc(self.fc_id_256_2_0)
        self._init_fc(self.fc_id_256_2_1)
        self._init_fc(self.fc_id_256_2_2)
        self.neck = args.neck
        if  self.neck == 'bnneck':
            self.bottleneck = nn.BatchNorm1d(256)
            self.bottleneck.bias.requires_grad_(False)  # no shift
            self.bottleneck.apply(weights_init_kaiming)

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

        x = self.backone(x)

        p1 = self.p1(x)
        p2 = self.p2(x)
        p3 = self.p3(x)

        zg_p1 = self.maxpool_zg_p1(p1)
        zg_p2 = self.maxpool_zg_p2(p2)
        zg_p3 = self.maxpool_zg_p3(p3)

        zp2 = self.maxpool_zp2(p2)
        z0_p2 = zp2[:, :, 0:1, :]
        z1_p2 = zp2[:, :, 1:2, :]

        zp3 = self.maxpool_zp3(p3)
        z0_p3 = zp3[:, :, 0:1, :]
        z1_p3 = zp3[:, :, 1:2, :]
        z2_p3 = zp3[:, :, 2:3, :]

        fg_p1 = self.reduction_0(zg_p1).squeeze(dim=3).squeeze(dim=2)
        fg_p2 = self.reduction_1(zg_p2).squeeze(dim=3).squeeze(dim=2)
        fg_p3 = self.reduction_2(zg_p3).squeeze(dim=3).squeeze(dim=2)
        f0_p2 = self.reduction_3(z0_p2).squeeze(dim=3).squeeze(dim=2)
        f1_p2 = self.reduction_4(z1_p2).squeeze(dim=3).squeeze(dim=2)
        f0_p3 = self.reduction_5(z0_p3).squeeze(dim=3).squeeze(dim=2)
        f1_p3 = self.reduction_6(z1_p3).squeeze(dim=3).squeeze(dim=2)
        f2_p3 = self.reduction_7(z2_p3).squeeze(dim=3).squeeze(dim=2)
        if  self.neck == 'bnneck':
            fg_p1_norm = self.bottleneck(fg_p1)  # normalize for angular softmax
            fg_p2_norm = self.bottleneck(fg_p2)
            fg_p3_norm = self.bottleneck(fg_p3)
            f0_p2_norm = self.bottleneck(f0_p2)
            f1_p2_norm = self.bottleneck(f1_p2)
            f0_p3_norm = self.bottleneck(f0_p3)
            f1_p3_norm = self.bottleneck(f1_p3)
            f2_p3_norm = self.bottleneck(f2_p3)
        else:
            fg_p1_norm = fg_p1
            fg_p2_norm = fg_p2
            fg_p3_norm = fg_p3
            f0_p2_norm = f0_p2
            f1_p2_norm = f1_p2
            f0_p3_norm = f0_p3
            f1_p3_norm = f1_p3
            f2_p3_norm = f2_p3

        '''
        l_p1 = self.fc_id_2048_0(zg_p1.squeeze(dim=3).squeeze(dim=2))
        l_p2 = self.fc_id_2048_1(zg_p2.squeeze(dim=3).squeeze(dim=2))
        l_p3 = self.fc_id_2048_2(zg_p3.squeeze(dim=3).squeeze(dim=2))
        '''
        l_p1 = self.fc_id_2048_0(fg_p1_norm)
        l_p2 = self.fc_id_2048_1(fg_p2_norm)
        l_p3 = self.fc_id_2048_2(fg_p3_norm)

        l0_p2 = self.fc_id_256_1_0(f0_p2_norm)
        l1_p2 = self.fc_id_256_1_1(f1_p2_norm)
        l0_p3 = self.fc_id_256_2_0(f0_p3_norm)
        l1_p3 = self.fc_id_256_2_1(f1_p3_norm)
        l2_p3 = self.fc_id_256_2_2(f2_p3_norm)

        predict = torch.cat([fg_p1_norm, fg_p2_norm, fg_p3_norm, f0_p2_norm, f1_p2_norm, f0_p3_norm, f1_p3_norm, f2_p3_norm], dim=1)

        return predict, fg_p1, fg_p2, fg_p3, f0_p2, f1_p2, f0_p3, f1_p3, f2_p3, l_p1, l_p2, l_p3, l0_p2, l1_p2, l0_p3, l1_p3, l2_p3

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

# class MGN_ibn(nn.Module):
#     def __init__(self, args):
#         super(MGN_ibn, self).__init__()
#         num_classes = args.num_classes
#         self.in_planes = 2048
#
#         self.backone = resnet50_ibn_a(args.last_stride)
#         if args.pretrain_choice == 'imagenet':
#             self.backone.load_param(args.pretrain_path)
#             print('Loading pretrained ImageNet model......')
#         else:
#             print('Random initial model......')
#
#
#
#         self.p1 = copy.deepcopy(self.backone)
#         self.p2 = copy.deepcopy(self.backone)
#         self.p3 = copy.deepcopy(self.backone)
#
#         if args.pool == 'max':
#             pool2d = nn.MaxPool2d
#         elif args.pool == 'avg':
#             pool2d = nn.AvgPool2d
#         else:
#             raise Exception()
#
#         self.maxpool_zg_p1 = pool2d(kernel_size=(24, 8))
#         self.maxpool_zg_p2 = pool2d(kernel_size=(24, 8))
#         self.maxpool_zg_p3 = pool2d(kernel_size=(24, 8))
#         self.maxpool_zp2 = pool2d(kernel_size=(12, 8))
#         self.maxpool_zp3 = pool2d(kernel_size=(8, 8))
#
#         reduction = nn.Sequential(nn.Conv2d(2048, args.feats, 1, bias=False), nn.BatchNorm2d(args.feats), nn.ReLU())
#
#         self._init_reduction(reduction)
#         self.reduction_0 = copy.deepcopy(reduction)
#         self.reduction_1 = copy.deepcopy(reduction)
#         self.reduction_2 = copy.deepcopy(reduction)
#         self.reduction_3 = copy.deepcopy(reduction)
#         self.reduction_4 = copy.deepcopy(reduction)
#         self.reduction_5 = copy.deepcopy(reduction)
#         self.reduction_6 = copy.deepcopy(reduction)
#         self.reduction_7 = copy.deepcopy(reduction)
#
#         # self.fc_id_2048_0 = nn.Linear(2048, num_classes)
#         self.fc_id_2048_0 = nn.Linear(args.feats, num_classes)
#         self.fc_id_2048_1 = nn.Linear(args.feats, num_classes)
#         self.fc_id_2048_2 = nn.Linear(args.feats, num_classes)
#
#         self.fc_id_256_1_0 = nn.Linear(args.feats, num_classes)
#         self.fc_id_256_1_1 = nn.Linear(args.feats, num_classes)
#         self.fc_id_256_2_0 = nn.Linear(args.feats, num_classes)
#         self.fc_id_256_2_1 = nn.Linear(args.feats, num_classes)
#         self.fc_id_256_2_2 = nn.Linear(args.feats, num_classes)
#
#         self._init_fc(self.fc_id_2048_0)
#         self._init_fc(self.fc_id_2048_1)
#         self._init_fc(self.fc_id_2048_2)
#
#         self._init_fc(self.fc_id_256_1_0)
#         self._init_fc(self.fc_id_256_1_1)
#         self._init_fc(self.fc_id_256_2_0)
#         self._init_fc(self.fc_id_256_2_1)
#         self._init_fc(self.fc_id_256_2_2)
#         self.neck = args.neck
#         self.neck_feat = args.neck_feat
#         if  self.neck == 'bnneck':
#             self.bottleneck = nn.BatchNorm1d(256)
#             self.bottleneck.bias.requires_grad_(False)  # no shift
#             self.bottleneck.apply(weights_init_kaiming)
#
#     @staticmethod
#     def _init_reduction(reduction):
#         # conv
#         nn.init.kaiming_normal_(reduction[0].weight, mode='fan_in')
#         # nn.init.constant_(reduction[0].bias, 0.)
#
#         # bn
#         nn.init.normal_(reduction[1].weight, mean=1., std=0.02)
#         nn.init.constant_(reduction[1].bias, 0.)
#
#     @staticmethod
#     def _init_fc(fc):
#         nn.init.kaiming_normal_(fc.weight, mode='fan_out')
#         # nn.init.normal_(fc.weight, std=0.001)
#         nn.init.constant_(fc.bias, 0.)
#
#     def forward(self, x):
#
#         p1 = self.p1(x)
#         p2 = self.p2(x)
#         p3 = self.p3(x)
#
#         zg_p1 = self.maxpool_zg_p1(p1)
#         zg_p2 = self.maxpool_zg_p2(p2)
#         zg_p3 = self.maxpool_zg_p3(p3)
#
#         zp2 = self.maxpool_zp2(p2)
#         z0_p2 = zp2[:, :, 0:1, :]
#         z1_p2 = zp2[:, :, 1:2, :]
#
#         zp3 = self.maxpool_zp3(p3)
#         z0_p3 = zp3[:, :, 0:1, :]
#         z1_p3 = zp3[:, :, 1:2, :]
#         z2_p3 = zp3[:, :, 2:3, :]
#
#         fg_p1 = self.reduction_0(zg_p1).squeeze(dim=3).squeeze(dim=2)
#         fg_p2 = self.reduction_1(zg_p2).squeeze(dim=3).squeeze(dim=2)
#         fg_p3 = self.reduction_2(zg_p3).squeeze(dim=3).squeeze(dim=2)
#         f0_p2 = self.reduction_3(z0_p2).squeeze(dim=3).squeeze(dim=2)
#         f1_p2 = self.reduction_4(z1_p2).squeeze(dim=3).squeeze(dim=2)
#         f0_p3 = self.reduction_5(z0_p3).squeeze(dim=3).squeeze(dim=2)
#         f1_p3 = self.reduction_6(z1_p3).squeeze(dim=3).squeeze(dim=2)
#         f2_p3 = self.reduction_7(z2_p3).squeeze(dim=3).squeeze(dim=2)
#         if  self.neck == 'bnneck':
#             fg_p1_norm = self.bottleneck(fg_p1)  # normalize for angular softmax
#             fg_p2_norm = self.bottleneck(fg_p2)
#             fg_p3_norm = self.bottleneck(fg_p3)
#             f0_p2_norm = self.bottleneck(f0_p2)
#             f1_p2_norm = self.bottleneck(f1_p2)
#             f0_p3_norm = self.bottleneck(f0_p3)
#             f1_p3_norm = self.bottleneck(f1_p3)
#             f2_p3_norm = self.bottleneck(f2_p3)
#         else:
#             fg_p1_norm = fg_p1
#             fg_p2_norm = fg_p2
#             fg_p3_norm = fg_p3
#             f0_p2_norm = f0_p2
#             f1_p2_norm = f1_p2
#             f0_p3_norm = f0_p3
#             f1_p3_norm = f1_p3
#             f2_p3_norm = f2_p3
#
#         '''
#         l_p1 = self.fc_id_2048_0(zg_p1.squeeze(dim=3).squeeze(dim=2))
#         l_p2 = self.fc_id_2048_1(zg_p2.squeeze(dim=3).squeeze(dim=2))
#         l_p3 = self.fc_id_2048_2(zg_p3.squeeze(dim=3).squeeze(dim=2))
#         '''
#         if self.training:
#             l_p1 = self.fc_id_2048_0(fg_p1_norm)
#             l_p2 = self.fc_id_2048_1(fg_p2_norm)
#             l_p3 = self.fc_id_2048_2(fg_p3_norm)
#
#             l0_p2 = self.fc_id_256_1_0(f0_p2_norm)
#             l1_p2 = self.fc_id_256_1_1(f1_p2_norm)
#             l0_p3 = self.fc_id_256_2_0(f0_p3_norm)
#             l1_p3 = self.fc_id_256_2_1(f1_p3_norm)
#             l2_p3 = self.fc_id_256_2_2(f2_p3_norm)
#
#
#
#             return fg_p1, fg_p2, fg_p3, f0_p2, f1_p2, f0_p3, f1_p3, f2_p3, l_p1, l_p2, l_p3, l0_p2, l1_p2, l0_p3, l1_p3, l2_p3
#         else:
#             if self.neck_feat == 'after':
#                 # print("Test with feature after BN")
#                 predict = torch.cat([fg_p1_norm, fg_p2_norm, fg_p3_norm, f0_p2_norm, f1_p2_norm, f0_p3_norm, f1_p3_norm, f2_p3_norm],dim=1)
#                 return predict
#             else:
#                 # print("Test with feature before BN")
#                 feature_final = torch.cat([fg_p1, fg_p2, fg_p3, f0_p2, f1_p2, f0_p3, f1_p3, f2_p3], dim=1)
#                 return feature_final
#
#
#
#
#     def save(self, apath, epoch, is_best=False):
#         target = self
#         torch.save(
#             target.state_dict(),
#             os.path.join(apath, 'model', 'model_{}.pt'.format(epoch))
#         )
#         if is_best:
#             torch.save(
#                 target.state_dict(),
#                 os.path.join(apath, 'model', 'model_best.pt')
#             )
#
#     def load_param(self, trained_path):
#         param_dict = torch.load(trained_path)
#         for i in param_dict:
#             self.state_dict()[i].copy_(param_dict[i])

class MGN_base(nn.Module):
    def __init__(self, args):
        super(MGN_base, self).__init__()
        num_classes = args.num_classes
        self.in_planes = 2048

        resnet = resnet50(pretrained=True)

        self.backone = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
        )
        self.res_g_conv3 = resnet.layer2
        self.res_p1_conv3 = copy.deepcopy(resnet.layer2)
        self.res_p2_conv3 = copy.deepcopy(resnet.layer2)
        self.trans21_g = TransformerBlock(512, 8)
        self.trans22_g = TransformerBlock(512, 8)
        self.trans21_p1 = TransformerBlock(512, 8)
        self.trans22_p1 = TransformerBlock(512, 8)
        self.cross21_p2 = CrossTransformerBlock(512, 8)
        self.cross22_p2 = CrossTransformerBlock(512, 8)

        self.res_g_conv4 = resnet.layer3
        self.res_p1_conv4 = copy.deepcopy(resnet.layer3)
        self.res_p2_conv4 = copy.deepcopy(resnet.layer3)
        self.trans31_g = TransformerBlock(1024, 16)
        self.trans32_g = TransformerBlock(1024, 16)
        self.trans33_g = TransformerBlock(1024, 16)
        self.trans31_p1 = TransformerBlock(1024, 16)
        self.trans32_p1 = TransformerBlock(1024, 16)
        self.trans33_p1 = TransformerBlock(1024, 16)
        self.cross31_p2 = CrossTransformerBlock(1024, 16)
        self.cross32_p2 = CrossTransformerBlock(1024, 16)
        self.cross33_p2 = CrossTransformerBlock(1024, 16)

        self.res_g_conv5 = resnet.layer4
        self.res_p1_conv5 = nn.Sequential(
            Bottleneck(1024, 512, downsample=nn.Sequential(nn.Conv2d(1024, 2048, 1, bias=False), nn.BatchNorm2d(2048))),
            Bottleneck(2048, 512),
            Bottleneck(2048, 512))
        self.res_p1_conv5.load_state_dict(resnet.layer4.state_dict())
        self.res_p2_conv5 = nn.Sequential(copy.deepcopy(self.res_p1_conv5))


        if args.pool == 'max':
            pool2d = nn.MaxPool2d
        elif args.pool == 'avg':
            pool2d = nn.AvgPool2d
        else:
            raise Exception()

        self.maxpool_zg_p1 = pool2d(kernel_size=(8, 4))
        self.maxpool_zg_p2 = pool2d(kernel_size=(16, 8))
        self.maxpool_zg_p3 = pool2d(kernel_size=(16, 8))

        reduction = nn.Sequential(nn.Conv2d(2048, 1024, 1, bias=False), nn.BatchNorm2d(1024), nn.ReLU())
        reduction1 = nn.Sequential(nn.Conv2d(2048, 512, 1, bias=False), nn.BatchNorm2d(512), nn.ReLU())

        self._init_reduction(reduction)
        self._init_reduction(reduction1)
        self.reduction_0 = copy.deepcopy(reduction)
        self.reduction_1 = copy.deepcopy(reduction1)
        self.reduction_2 = copy.deepcopy(reduction1)

        # self.fc_id_2048_0 = nn.Linear(2048, num_classes)
        self.fc_id_2048_0 = nn.Linear(1024, num_classes)
        self.fc_id_2048_1 = nn.Linear(512, num_classes)
        self.fc_id_2048_2 = nn.Linear(512, num_classes)

        self._init_fc(self.fc_id_2048_0)
        self._init_fc(self.fc_id_2048_1)
        self._init_fc(self.fc_id_2048_2)
        self.neck_feat = args.neck_feat
        self.neck = args.neck
        if  self.neck == 'bnneck':
            self.bottleneck = nn.BatchNorm1d(1024)
            self.bottleneck.bias.requires_grad_(False)  # no shift
            self.bottleneck.apply(weights_init_kaiming)
            self.bottleneck1 = nn.BatchNorm1d(512)
            self.bottleneck1.bias.requires_grad_(False)  # no shift
            self.bottleneck1.apply(weights_init_kaiming)
            self.bottleneck2 = copy.deepcopy(self.bottleneck1)

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

        x = self.backone(x)
        g, p1, p2 = x, x, x
        for i in range(len(self.res_g_conv3)):
            g = self.res_g_conv3[i](g)
            p1 = self.res_p1_conv3[i](p1)
            p2 = self.res_p2_conv3[i](p2)
            if i == 2:
                g = self.trans21_g(g)
                p1 = self.trans21_p1(p1)
                p2 = self.cross21_p2(p1,p2)
            if i == 3:
                g = self.trans22_g(g)
                p1 = self.trans22_p1(p1)
                p2 = self.cross22_p2(p1, p2)

        for i in range(len(self.res_g_conv4)):
            g = self.res_g_conv4[i](g)
            p1 = self.res_p1_conv4[i](p1)
            p2 = self.res_p2_conv4[i](p2)
            if i == 3:
                g = self.trans31_g(g)
                p1 = self.trans31_p1(p1)
                p2 = self.cross31_p2(p1, p2)
            if i == 4:
                g = self.trans32_g(g)
                p1 = self.trans32_p1(p1)
                p2 = self.cross32_p2(p1, p2)
            if i == 5:
                g = self.trans33_g(g)
                p1 = self.trans33_p1(p1)
                p2 = self.cross33_p2(p1, p2)

        g = self.res_g_conv5(g)
        p1 = self.res_p1_conv5(p1)
        p2 = self.res_p2_conv5(p2)

        zg_g = self.maxpool_zg_p1(g)
        zg_p1 = self.maxpool_zg_p2(p1)
        zg_p2 = self.maxpool_zg_p3(p2)

        fg_g = self.reduction_0(zg_g).squeeze(dim=3).squeeze(dim=2)
        fg_p1 = self.reduction_1(zg_p1).squeeze(dim=3).squeeze(dim=2)
        fg_p2 = self.reduction_2(zg_p2).squeeze(dim=3).squeeze(dim=2)
        if  self.neck == 'bnneck':
            fg_g_norm = self.bottleneck(fg_g)  # normalize for angular softmax
            fg_p1_norm = self.bottleneck1(fg_p1)
            fg_p2_norm = self.bottleneck2(fg_p2)
        else:
            fg_g_norm = fg_g
            fg_p1_norm = fg_p1
            fg_p2_norm = fg_p2

        if self.training:
            l_g = self.fc_id_2048_0(fg_g_norm)
            l_p1 = self.fc_id_2048_1(fg_p1_norm)
            l_p2 = self.fc_id_2048_2(fg_p2_norm)

            return fg_g, fg_p1, fg_p2, l_g, l_p1, l_p2
        else:
            if self.neck_feat == 'after':
                # print("Test with feature after BN")
                feature_final = torch.cat([fg_g_norm, fg_p1_norm, fg_p2_norm], dim=1)
                return feature_final
            else:
                # print("Test with feature before BN")
                feature_final = torch.cat([fg_g, fg_p1, fg_p2], dim=1)
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
if __name__ == '__main__':
    from running_mod import args
    from torchsummary import summary

    args.num_classes = 751
    args.pretrain_choice = None
    model = MGN_base(args)
    #print(model)
    summary(model.cuda(), (3, 384, 128), 64)
