from torch.utils.data import DataLoader
from icesat2cept.utils.registroy import Registry


DATASETS = Registry("datasets")


def build_dataset(cfg):
    """Build datasets."""
    return DATASETS.build(cfg)

def build_dataloader(cfg):
    dataset = build_dataset(cfg.dataset)
    dataloader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers)
    return dataloader