# encoding: utf-8
import torchvision.transforms as T
from loaders import init_dataset, ImageDataset, ImagePath
from torch.utils.data import DataLoader


from torch.utils.data.sampler import Sampler
from collections import defaultdict
import copy
import random
import numpy as np
class RandomIdentitySampler(Sampler):
    """
    Randomly sample N identities, then for each identity,
    randomly sample K instances, therefore batch size is N*K.
    Args:
    - data_source (list): list of (img_path, pid, camid).
    - num_instances (int): number of instances per identity in a batch.
    - batch_size (int): number of examples in a batch.
    """

    def __init__(self, data_source, batch_size, num_instances):  # dataset.train
        self.data_source = data_source
        self.batch_size = batch_size
        self.num_instances = num_instances
        self.num_pids_per_batch = self.batch_size // self.num_instances
        self.index_dic = defaultdict(list)
        for index, (_, pid, _) in enumerate(self.data_source):
            self.index_dic[pid].append(index)
        self.pids = list(self.index_dic.keys())

        # estimate number of examples in an epoch
        self.length = 0
        for pid in self.pids:
            idxs = self.index_dic[pid]
            num = len(idxs)
            if num < self.num_instances:
                num = self.num_instances
            self.length += num - num % self.num_instances

    def __iter__(self):
        batch_idxs_dict = defaultdict(list)

        for pid in self.pids:
            idxs = copy.deepcopy(self.index_dic[pid])
            if len(idxs) < self.num_instances:
                idxs = np.random.choice(idxs, size=self.num_instances, replace=True)
            random.shuffle(idxs)
            batch_idxs = []
            for idx in idxs:
                batch_idxs.append(idx)
                if len(batch_idxs) == self.num_instances:
                    batch_idxs_dict[pid].append(batch_idxs)
                    batch_idxs = []

        avai_pids = copy.deepcopy(self.pids)
        final_idxs = []

        while len(avai_pids) >= self.num_pids_per_batch:
            selected_pids = random.sample(avai_pids, self.num_pids_per_batch)
            for pid in selected_pids:
                batch_idxs = batch_idxs_dict[pid].pop(0)
                final_idxs.extend(batch_idxs)
                if len(batch_idxs_dict[pid]) == 0:
                    avai_pids.remove(pid)

        self.length = len(final_idxs)
        return iter(final_idxs)

    def __len__(self):
        return self.length


import math
class RandomErasing(object):
    """ Randomly selects a rectangle region in an image and erases its pixels.
        'Random Erasing Data Augmentation' by Zhong et al.
        See https://arxiv.org/pdf/1708.04896.pdf
    Args:
         probability: The probability that the Random Erasing operation will be performed.
         sl: Minimum proportion of erased area against input image.
         sh: Maximum proportion of erased area against input image.
         r1: Minimum aspect ratio of erased area.
         mean: Erasing value.
    """

    def __init__(self, probability=0.5, sl=0.02, sh=0.4, r1=0.3, mean=(0.4914, 0.4822, 0.4465)):
        self.probability = probability
        self.mean = mean
        self.sl = sl
        self.sh = sh
        self.r1 = r1

    def __call__(self, img):

        if random.uniform(0, 1) >= self.probability:
            return img

        for attempt in range(100):
            area = img.size()[1] * img.size()[2]

            target_area = random.uniform(self.sl, self.sh) * area
            aspect_ratio = random.uniform(self.r1, 1 / self.r1)

            h = int(round(math.sqrt(target_area * aspect_ratio)))
            w = int(round(math.sqrt(target_area / aspect_ratio)))

            if w < img.size()[2] and h < img.size()[1]:
                x1 = random.randint(0, img.size()[1] - h)
                y1 = random.randint(0, img.size()[2] - w)
                if img.size()[0] == 3:
                    img[0, x1:x1 + h, y1:y1 + w] = self.mean[0]
                    img[1, x1:x1 + h, y1:y1 + w] = self.mean[1]
                    img[2, x1:x1 + h, y1:y1 + w] = self.mean[2]
                else:
                    img[0, x1:x1 + h, y1:y1 + w] = self.mean[0]
                return img

        return img

import torch
def train_collate_fn(batch):
    imgs, pids, _, _= zip(*batch)  # imgs_and_weights tuple
    pids = torch.tensor(pids, dtype=torch.int64)
    return torch.stack(imgs, dim=0), pids  # 保留图像和标签以及权重，重新打包
def val_collate_fn(batch):
    imgs, pids, camids, _ = zip(*batch)
    return torch.stack(imgs, dim=0), pids, camids
from utils.half_crop import HalfCrop
def make_data_loader(args):
    train_transforms = T.Compose([
            T.Resize((args.height,args.width)),
            T.RandomHorizontalFlip(),
            T.Pad(10),
            T.RandomCrop((args.height, args.width)),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            RandomErasing(probability=args.random_erasing_probability, mean=(0.0, 0.0, 0.0))
        ])
    #print('train_transforms', train_transforms)
    val_transforms = T.Compose([
            T.Resize((args.height,args.width)),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    dataset = init_dataset(args.dataset_name, root=args.datadir)  # 形成对应数据集的对象
    train_set = ImageDataset(dataset.train, train_transforms)  # 含标签数据集,只有训练集标签被重置为0-750
    query_set = ImageDataset(dataset.query, val_transforms)
    gallery_set = ImageDataset(dataset.gallery, val_transforms)
    if dataset.train is not None: # for occluded reid dataset
        if args.loss.find('Triplet') == -1: # 无TripletLoss时采样
            train_loader = DataLoader(
                train_set, batch_size=args.batchid * args.batchimage,
                shuffle = True, num_workers=args.nThread, collate_fn=train_collate_fn,pin_memory=True)
        else:
            print("三元组随机采样！")
            train_loader = DataLoader(  # 最终输出的train_loader含图像数组、标签、权重
                train_set,
                sampler=RandomIdentitySampler(dataset.train, args.batchid * args.batchimage, args.batchimage),
                batch_size=args.batchid * args.batchimage,
                num_workers=args.nThread,
                collate_fn=train_collate_fn,pin_memory=True
            )
    else:train_loader = None

    gallery_loader = DataLoader(
        gallery_set, batch_size=args.batchtest, shuffle=False, num_workers=args.nThread,
        collate_fn=val_collate_fn,pin_memory=True
    )
    query_loader = DataLoader(
        query_set, batch_size=args.batchtest, shuffle=False, num_workers=args.nThread,
        collate_fn=val_collate_fn,pin_memory=True
    )
    return train_loader, query_loader, gallery_loader, dataset.num_train_pids

def get_paths_for_retrieve(args):
    dataset = init_dataset(args.dataset_name, root=args.datadir)  # 形成对应数据集的对象
    query_paths = ImagePath(dataset.query)
    gallery_paths = ImagePath(dataset.gallery)

    return query_paths, gallery_paths
