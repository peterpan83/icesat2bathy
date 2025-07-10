from icesat2cept.engines.builder import TRAINERS
from icesat2cept.engines.launch import launch

from icesat2cept.engines.default import (default_setup,
                                         default_config_parser
)

def main_worker(config):
    config = default_setup(config)
    trainer = TRAINERS.build(dict(type=config.trainer.type, config=config))
    trainer.train()


def main(cfg_file):
    # args = default_argument_parser().parse_args()
    cfg = default_config_parser(file_path=cfg_file, options=None)
    launch(
        main_worker,
        num_gpus_per_machine=1,
        num_machines=1,
        machine_rank=0,
        dist_url=None,
        cfg=(cfg,),
    )