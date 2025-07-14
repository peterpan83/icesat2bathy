from addict import Dict
from typing import Optional
import torch

from icesat2cept.utils.point_ops import fps_1d

try:
    from knn_cuda import KNN
    print('using GPU KNN')
except ImportError:
    from icesat2cept.utils.point_ops import KNN
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

        coords = self['coords']#.to(device)
        feats = self['feature']#.to(device)
        labels = self['label']#.to(device)

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

    def serilization(self, descending=Optional[bool]):
        if descending is not None:
            ### use descending order or acsending order
            self.center_index, indices = torch.sort(self.center_index, dim=1) if not descending else torch.sort(self.center_index, descending=True, dim=1)
            indices = indices.view(-1)
            # self.center_coords = self.center_coords.[indices]
            for key in self.neighborhood.keys():
                dim = self.neighborhood[key].shape
                if len(dim) == 4:
                    self.neighborhood[key] = self.neighborhood[key].flatten(0, 1)[indices].view(dim[0], dim[1], dim[2], dim[3])
                elif len(dim) == 3:
                    self.neighborhood[key] = self.neighborhood[key].flatten(0, 1)[indices].view(dim[0], dim[1], dim[2])
                elif len(dim) == 2:
                    self.neighborhood[key] = self.neighborhood[key].flatten(0, 1)[indices].view(dim[0], dim[1])
        else:
            ### use both order
            pass

    def move_to_device(self, data, device):
        if isinstance(data, torch.Tensor):
            return data.to(device)
        elif isinstance(data, dict):
            return {k: self.move_to_device(v, device) for k, v in data.items()}
        elif isinstance(data, list):
            return [self.move_to_device(v, device) for v in data]
        elif isinstance(data, tuple):
            return tuple(self.move_to_device(v, device) for v in data)
        else:
            return data








if __name__ == '__main__':
    points = {'a':range(1,10), 'b':range(1,10), 'c':range(1,10)}
    icesat = IceSatDict(**points)
    icesat.d = 10
    print(icesat)
