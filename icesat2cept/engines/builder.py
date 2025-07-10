'''
bud
'''

from icesat2cept.utils.registroy import Registry
from icesat2cept.utils.logger import get_root_logger


################################## trainer #########
TRAINERS = Registry('trainer')
def build_trainer(cfg):
    return TRAINERS.build(cfg)


####################################### optimizers #########
from torch.optim import (
    SGD,
    Adam,
    AdamW
)
OPTIMIZERS = Registry('optimizers')
OPTIMIZERS.register_module(module=AdamW, name='AdamW')
def build_optimizer(cfg, model, param_dicts=None):
    if param_dicts is None:
        cfg.params = model.parameters()
    else:
        cfg.params = [dict(names=[], params=[], lr=cfg.lr)]
        for i in range(len(param_dicts)):
            param_group = dict(names=[], params=[])
            if "lr" in param_dicts[i].keys():
                param_group["lr"] = param_dicts[i].lr
            if "momentum" in param_dicts[i].keys():
                param_group["momentum"] = param_dicts[i].momentum
            if "weight_decay" in param_dicts[i].keys():
                param_group["weight_decay"] = param_dicts[i].weight_decay
            cfg.params.append(param_group)

        for n, p in model.named_parameters():
            flag = False
            for i in range(len(param_dicts)):
                if param_dicts[i].keyword in n:
                    cfg.params[i + 1]["names"].append(n)
                    cfg.params[i + 1]["params"].append(p)
                    flag = True
                    break
            if not flag:
                cfg.params[0]["names"].append(n)
                cfg.params[0]["params"].append(p)

        logger = get_root_logger()
        for i in range(len(cfg.params)):
            param_names = cfg.params[i].pop("names")
            message = ""
            for key in cfg.params[i].keys():
                if key != "params":
                    message += f" {key}: {cfg.params[i][key]};"
            logger.info(f"Params Group {i+1} -{message} Params: {param_names}.")
    return OPTIMIZERS.build(cfg=cfg)

############################# scheduler ###########################
from torch.optim.lr_scheduler import (
    OneCycleLR
)

SCHEDULERS = Registry('schedulers')
SCHEDULERS.register_module(module=OneCycleLR, name='OneCycleLR')

def build_schedulers(cfg,optimizer):
    cfg.optimizer = optimizer
    return SCHEDULERS.build(cfg)



###############################Hooks##############
HOOKS =Registry('hooks')

def build_hooks(hooks_list:list):
    hooks = []
    for hook_cfg in hooks_list:
        hooks.append(HOOKS.build(hook_cfg))
    return hooks


#############################Criteria##################
LOSSES = Registry(name='losses')
class Criteria(object):
    def __init__(self, cfg=None):
        self.cfg = cfg if cfg is not None else []
        self.criteria = []
        for loss_cfg in self.cfg:
            self.criteria.append(LOSSES.build(cfg=loss_cfg))

    def __call__(self, pred, target):
        if len(self.criteria) == 0:
            # loss computation occur in model
            return pred
        loss = 0
        for c in self.criteria:
            loss += c(pred, target)
        return loss


def build_criteria(cfg):
    return Criteria(cfg)





