from __future__ import absolute_import
import warnings

from .market1501 import Market1501
from .msmt17 import MSMT17
from .dukemtmcreid import DukeMTMCreID
from .personx import PersonX
from .cuhk03 import CUHK03NPD


__factory = {
    'market1501': Market1501,
    'msmt17': MSMT17,
    'dukemtmcreid': DukeMTMCreID,
    'personx': PersonX,
    'cuhk03-np-d': CUHK03NPD,
}


def names():
    return sorted(__factory.keys())


def create(name, root, *args, **kwargs):

    if name not in __factory:
        raise KeyError("Unknown dataset:", name)
    return __factory[name](root, *args, **kwargs)


def get_dataset(name, root, *args, **kwargs):
    warnings.warn("get_dataset is deprecated. Use create instead.")
    return create(name, root, *args, **kwargs)
