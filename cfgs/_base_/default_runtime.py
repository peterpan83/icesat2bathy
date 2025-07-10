weight = None  # path to model weight
resume = False  # whether to resume training process
evaluate = True  # evaluate after each epoch training process
test_only = False  # test process

seed = None  # train process will init a random seed and record
save_path = "/home/yan/WorkSpace/developments/icesatcept"
num_worker = 1  # total worker in all gpu
batch_size = 4  # total batch size in all gpu
batch_size_val = None  # auto adapt to bs 1 for each gpu
batch_size_test = None  # auto adapt to bs 1 for each gpu
epoch = 100  # total epoch, data loop = epoch // eval_epoch
eval_epoch = 100  # sche total eval & checkpoint epoch
clip_grad = None  # disable with None, enable with a float

sync_bn = False
enable_amp = False
empty_cache = False
empty_cache_per_epoch = False
find_unused_parameters = False

param_dicts = None  # example: param_dicts = [dict(keyword="block", lr_scale=0.1)]

hooks = [
        dict(type='IterationTimer', warmup_iter=2),
        dict(type='InformationWriter')
    ]

trainer = dict(
    type='DefaultTrainer',
)

optimizer = dict(type="AdamW", lr=0.0006, weight_decay=0.05)
scheduler = dict(
    type="OneCycleLR",
    max_lr=0.006,
    pct_start=0.05,
    anneal_strategy="cos",
    div_factor=10.0,
    final_div_factor=1000.0,
    total_steps=None,
    cycle_momentum=True,
    base_momentum=0.85,
    max_momentum=0.95,
    three_phase=False,
    last_epoch=-1,
    epochs = epoch,  ## will be overwriten as
    steps_per_epoch = 0 ### will be overwriten as len(data_loader)
)
