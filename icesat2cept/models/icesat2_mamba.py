import torch
import torch.nn as nn


from icesat2cept.utils.logger import get_root_logger

_logger = get_root_logger()


class MaskMamba_1D(nn.Module):

    def __init__(self, config, **kwargs):
        super(MaskMamba_1D, self).__init__()
        super().__init__()
        self.config = config
        self.mask_ratio = config.transformer_config.mask_ratio
        self.trans_dim = config.transformer_config.trans_dim
        self.depth = config.transformer_config.depth
        self.num_heads = config.transformer_config.num_heads

        # embedding
        self.encoder_dims = config.transformer_config.encoder_dims
        self.encoder = Encoder(encoder_channel=self.encoder_dims)

        self.mask_type = config.transformer_config.mask_type
        self.pos_embed = nn.Sequential(
            nn.Linear(1, 128),
            nn.GELU(),
            nn.Linear(128, self.trans_dim),
        )



class IcesatMAEMamba(nn.Module):
    def __init__(self, config, **kwargs):
        super(IcesatMAEMamba, self).__init__()


    def forward(self, x):
        pass
