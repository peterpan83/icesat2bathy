from addict import Dict
import torch

from .point_ops import fps_1d

try:
    from knn_cude import KNN
    print('using GPU KNN')
except ImportError:
    from .point_ops import KNN
    print('using CPU KNN')

class IceSatDict(Dict):
    '''
    coords: along track distance, 1D
    height: Geoid_Corrected_Ortho_Height
    feature: IceSat features
    '''

    def __init__(self, *args,**kwargs):
        super(IceSatDict, self).__init__(*args,**kwargs)

    def group(self, group_number, group_size):
        coords = self['coords']
        feats = self['feature']
        labels = self['label']

        B, N, D = coords.shape  # batch size, number of points, dimension of coords
        G, M = group_number, group_size
        # fps the centers out
        center_index, center_coords = fps_1d(coords, G)  # B G D
        # knn to get the neighborhood
        # idx = knn_query(xyz, center, self.group_size)  # B G M
        knn = KNN(M, transpose_mode=True)
        _, idx = knn(coords, center_coords)  # B G M
        assert idx.size(1) == G
        assert idx.size(2) == M
        idx_base = torch.arange(0, B, device=coords.device).view(-1, 1, 1) * N
        idx = idx + idx_base
        idx = idx.view(-1)

        neighborhood = {}

        neighborhood['coords'] = coords.view(B * N, -1)[idx, :].view(B, G, M, D).contiguous()
        neighborhood['coords'] = neighborhood['coords'] - center_coords.unsqueeze(2)

        neighborhood['feature'] = feats.view(B * N, -1)[idx, :].view(B, G, M, feats.shape[-1]).contiguous()
        neighborhood['label'] = labels.view(B * N)[idx].view(B, G, M).contiguous()

        self.neighborhood = neighborhood
        self.center_index = center_index
        self.center_coords = center_coords



if __name__ == '__main__':
    points = {'a':range(1,10), 'b':range(1,10), 'c':range(1,10)}
    icesat = IceSatDict(**points)
    icesat.d = 10
    print(icesat)
