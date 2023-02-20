# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""
# from .cuhk03 import CUHK03
from .dukemtmcreid import DukeMTMCreID, Occluded_Duke
from .market1501 import Market1501
from .msmt17 import MSMT17
from .veri import VeRi
from .occluded_reid import OccludedReID
from .partial_ilids import PartialILIDS
from .partial_reid import PartialREID
from .dataset_loader import ImageDataset, ImagePath

__factory = {
    'market1501': Market1501,
    # 'cuhk03': CUHK03,
    'dukemtmc': DukeMTMCreID,
    'occluded_dukemtmc': Occluded_Duke,
    'occluded_reid': OccludedReID,
    'partial_ilids': PartialILIDS,
    'partial_reid': PartialREID,
    'msmt17': MSMT17,
    'veri': VeRi,
}


def get_names():
    return __factory.keys()


def init_dataset(name, *args, **kwargs):
    if name not in __factory.keys():
        raise KeyError("Unknown loaders: {}".format(name))
    return __factory[name](*args, **kwargs)
