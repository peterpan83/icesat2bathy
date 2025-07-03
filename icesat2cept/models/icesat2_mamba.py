import torch
import torch.nn as nn
from timm.models.layers import trunc_normal_, DropPath

from .builder import MODELS
from icesat2cept.utils.logger import get_root_logger
from icesat2cept.utils.structures import IceSatDict

_logger = get_root_logger()

class PosEmebeding(nn.Module):

    def __init__(self, emb_dims=128, layernorm=True):
        super(PosEmebeding, self).__init__()
        self.pos_emb = nn.Sequential(
            nn.Linear(1, 128),
            nn.GELU(),
            nn.Linear(128, emb_dims),
            nn.Identity() if not layernorm else nn.LayerNorm(emb_dims),
        )

    def forward(self, x):
        return self.pos_emb(x)

class FeatureEmbeding(nn.Module):
    '''
    embeding the features of points in each neiborhood in groups

    dimension: b, g, n, c(feature for each point) --> b, g, e(embed dimension)

    '''
    def __init__(self, feat_dims=1, emb_dims=128):
        super(FeatureEmbeding, self).__init__()
        self.feat_dims = feat_dims
        self.emb_dims = emb_dims

        self.first_conv = nn.Sequential(
            nn.Conv1d(self.feat_dims, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 256, 1),
        )

        self.second_conv = nn.Sequential(
            nn.Conv1d(512, 512, 1),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, self.emb_dims, 1)
        )

        self.pos_emb = PosEmebeding(emb_dims=256, layernorm=True)


    def forward(self, icesat: IceSatDict):
        feats = icesat.neighborhood['feature']

        b, g, n, c = feats.shape
        feats_reshape = feats.reshape(b * g, n, self.feat_dims) ## bg, n, c
        _feats = self.first_conv(feats_reshape.transpose(1, 2))  ## bg, 256, n

        relative_pos = icesat.neighborhood['coords']  ## b,g,n,1
        # _relative_pos = self.first_conv(relative_pos.reshape(b * g, n, self.feat_dims).transpose(1, 2))
        _relative_pos = self.pos_emb(relative_pos.reshape(b * g, n, 1)).transpose(1, 2) ## bg, 256, n

        ### add relative position embeding
        _feats += _relative_pos ## bg, 256, n

        _feats_glb = torch.max(_feats, dim=2, keepdim=True)[0]  ## bg, 256, 1
        _feats = torch.cat([_feats, _feats_glb.expand(-1, -1, n)], dim=1) ##bg, 512, n
        _feats = self.second_conv(_feats) ## bg, e, n
        _feats_glb = torch.max(_feats, dim=2, keepdim=False)[0] ## bg, e
        _feats  = _feats_glb.reshape(b, g, self.emb_dims) ## b, g, e

        icesat.neighborhood['emb_feature'] = _feats  ## b, g, e
        return icesat


class MaskMamba_1D(nn.Module):

    def __init__(self,
                 mask_ratio = 0.5,
                 trans_dim = 384,
                 depth =12,
                 num_heads = 6,
                 encoder_dims = 384,
                 mask_type = 'rand',
                 **kwargs):
        super(MaskMamba_1D, self).__init__()
        super().__init__()
        self.mask_ratio = mask_ratio
        self.trans_dim = trans_dim
        self.depth = depth
        self.num_heads = num_heads

        self.mask_type = mask_type
        self.pos_embed = PosEmebeding(emb_dims=self.trans_dim,layernorm=True)

        self.apply(self._init_weights)


    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv1d):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def _mask_center_rand(self, center):
        B, G, _ = center.shape ## B, G, 1
        # skip the mask
        if self.mask_ratio == 0:
            return torch.zeros((B, G), dtype=torch.bool, device=center.device)

        self.num_mask = int(self.mask_ratio * G)

        # Generate random scores and take top `num_mask` positions for masking
        rand_scores = torch.rand(B, G, device=center.device)
        _, indices = torch.topk(rand_scores, self.num_mask, dim=1, largest=True)

        # Create a mask initialized to False
        mask = torch.zeros((B, G), dtype=torch.bool, device=center.device)
        # Scatter True into the mask at the top-k indices
        mask.scatter_(1, indices, True)

        return mask  # B G

    def forward(self, icesat: IceSatDict):

        feats_emb = icesat.neighborhood['emb_feature']
        center_coords = icesat.center_coords
        B, G, E = feats_emb.shape

        if self.mask_type == 'rand':
            bool_masked_pos = self._mask_center_rand(center_coords)

        feats_emb_vis = feats_emb[~bool_masked_pos].reshape(B, -1, E)
        center_coords_vis = center_coords[~bool_masked_pos].reshape(B, -1, 1)

        ### add positional embeding
        pos_emb = self.pos_embed(center_coords_vis)

        return icesat


@MODELS.register_module()
class IcesatMAEMamba(nn.Module):
    def __init__(self,
                group_number = 1000,
                group_size = 20,
                mask_ratio=0.5,
                mask_type='rand',
                trans_dim = 384,
                encoder_dims = 384,
                depth = 12,
                drop_path_rate = 0.1,
                num_heads= 6,
                decoder_depth = 4,
                decoder_num_heads = 6, **kwargs):
        super(IcesatMAEMamba, self).__init__()
        self.mask_ratio = mask_ratio
        self.mask_type = mask_type
        self.trans_dim = trans_dim
        self.depth = depth
        self.encoder_dims = encoder_dims
        self.drop_path_rate = drop_path_rate
        self.num_heads = num_heads
        self.decoder_depth = decoder_depth
        self.decoder_num_heads = decoder_num_heads

        self.group_size = group_size
        self.group_num = group_number

        self.encoder = MaskMamba_1D(mask_ratio=self.mask_ratio,
                                    mask_type=self.mask_type,
                                    trans_dim=self.trans_dim,
                                    depth=self.depth,
                                    num_heads=self.num_heads,
                                    decoder_depth=self.decoder_depth,
                                    )

        self.embeding_dims = encoder_dims
        self.embeding = FeatureEmbeding(feat_dims=1, emb_dims=self.embeding_dims)

    def forward(self, data_dict):
        icesat = IceSatDict(data_dict)
        icesat.group(group_size=self.group_size, group_number=self.group_num)
        icesat = self.embeding(icesat)

        icesat = self.encoder(icesat)
        return icesat
