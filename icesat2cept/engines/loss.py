from .builder import LOSSES

from torch.nn import MSELoss

LOSSES.register_module(module=MSELoss, name='mseloss')
