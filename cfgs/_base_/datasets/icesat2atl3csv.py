
dataset_type = "Icesat2ATL3"
data_root = "/home/yan/WorkSpace/dataset/icesat2cept"
num_classes = 3
names = [
    "sea surface",
    "sea floor",
    "others"
]
train_dataset = dict(
    type = dataset_type,
    data_root = data_root,
    split = "train",
    filter_str = "*.csv",
    test_mode = False,
    transform = [
        dict(type='ToTensor'),
    ]
)

train_dataloader = dict(
    batch_size = 2,
    num_workers = 0,
    persistent_workers = True,
    sampler = dict(type='InfiniteSampler', shuffle=True),
    dataset = train_dataset,
)



