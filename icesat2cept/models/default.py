import torch
import torch.nn as nn

from icesat2cept.engines.structures import IceSatDict
from icesat2cept.engines.builder import build_criteria

from .builder import MODELS, build_model

@MODELS.register_module()
class DefaultMAE(nn.Module):
    '''
    default masked encoder
    '''
    def __init__(self,
                 backbone,
                 criteria
                 ):

        super(DefaultMAE, self).__init__()
        self.backbone = build_model(backbone)
        self.criteria = build_criteria(criteria)

    def forward(self, icesat_dict):
        icesat = IceSatDict(icesat_dict)
        icesat = self.backbone(icesat)

        mask = icesat.neighborhood['masked_center']
        gt_heigh = icesat.neighborhood['feature']

        # print(icesat['recover_height_mask'].shape, gt_heigh[mask].shape)
        loss = self.criteria(icesat['recover_height_mask'], gt_heigh[mask])
        return dict(loss=loss)
        # self.criteria()









