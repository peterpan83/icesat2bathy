from functools import partial
import sys, os, weakref
if sys.version_info >= (3, 10):
    from collections.abc import Iterator
else:
    from collections import Iterator

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from icesat2cept.models.builder import build_model
from icesat2cept.datasets.builder import build_dataset
from icesat2cept.utils.logger import get_root_logger
from icesat2cept.utils.event import EventStorage,ExceptionWriter

import icesat2cept.utils.comm as comm

from .default import create_ddp_model, worker_init_fn

from .builder import TRAINERS
from .builder import (build_optimizer,
                      build_hooks,
                      build_schedulers
                      )
from .hooks import HookBase

class TrainerBase(object):
    def __init__(self):
        self.hooks = []
        self.current_epoch = 0
        self.start_epoch = 0
        self.max_epoch = 0

        self.comm_info = dict()
        self.data_iterator: Iterator = enumerate([])
        self.storage: EventStorage = None
        self.writer:SummaryWriter = None

    def train(self):
        with EventStorage() as self.storage:
            # => before train
            self.before_train()
            for self.epoch in range(self.start_epoch, self.max_epoch):
                # => before epoch
                self.before_epoch()
                # => run_epoch
                for (
                    self.comm_info["iter"],
                    self.comm_info["input_dict"],
                ) in self.data_iterator:
                    # => before_step
                    self.before_step()
                    # => run_step
                    self.run_step()
                    # => after_step
                    self.after_step()
                # => after epoch
                self.after_epoch()
            # => after train
            self.after_train()

    def before_train(self):
        for h in self.hooks:
            h.before_train()

    def before_epoch(self):
        for h in self.hooks:
            h.before_epoch()

    def before_step(self):
        for h in self.hooks:
            h.before_step()

    def run_step(self):
        raise NotImplementedError

    def after_step(self):
        for h in self.hooks:
            h.after_step()

    def after_epoch(self):
        for h in self.hooks:
            h.after_epoch()
        self.storage.reset_histories()

    def after_train(self):
        # Sync GPU before running train hooks
        comm.synchronize()
        for h in self.hooks:
            h.after_train()
        if comm.is_main_process():
            self.writer.close()

    def register_hooks(self, hooks) -> None:
        hooks = build_hooks(hooks)
        for h in hooks:
            assert isinstance(h, HookBase)
            # To avoid circular reference, hooks and trainer cannot own each other.
            # This normally does not matter, but will cause memory leak if the
            # involved objects contain __del__:
            # See http://engineering.hearsaysocial.com/2013/06/16/circular-references-in-python/
            h.trainer = weakref.proxy(self)
        self.hooks.extend(hooks)

@TRAINERS.register_module()
class DefaultTrainer(TrainerBase):

    def __init__(self, config):
        super(DefaultTrainer, self).__init__()

        self.config = config
        self.save_path = config.save_path
        self.max_epoch = config.eval_epoch
        ### logger
        self.logger = get_root_logger()
        self.logger.set_logdir(self.save_path)

        ### build dataloader
        self.dataloader_train = self.build_train_loader()
        self.dataloader_val = self.build_val_loader()
        self.dataloader_test = None

        ### build model, optimizer and scheduler
        self.model = self.build_model()
        self.optimizer = self.build_optimizer()
        self.scheduler = self.build_scheduler()
        self.scaler = self.build_scaler()

        self.writer = self.build_writer()


        self.register_hooks(self.config.hooks)

    def build_writer(self):
        writer = SummaryWriter(self.config.save_path) if comm.is_main_process() else None
        self.logger.info(f"Tensorboard writer logging dir: {self.config.save_path}")
        return writer
    def build_scaler(self):
        scaler = torch.cuda.amp.GradScaler() if self.config.enable_amp else None
        return scaler
    def build_model(self):
        config = self.config.model
        model = build_model(config)

        n_trainable = sum([p.numel() for p in model.parameters() if p.requires_grad])
        self.logger.info(f'n_trainable: {n_trainable}')
        if torch.cuda.is_available():
            if self.config.sync_bn:
                model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
            model = create_ddp_model(model=model.cuda(), fp16_compression=True)

        return model

    def build_train_loader(self):
        dataset = build_dataset(self.config.train_dataset)

        if comm.get_world_size() > 1:
            train_sampler = torch.utils.data.distributed.DistributedSampler(dataset)
        else:
            train_sampler = None

        init_fn = (
            partial(
                worker_init_fn,
                num_workers=self.config.num_worker_per_gpu,
                rank=comm.get_rank(),
                seed=self.config.seed,
            )
            if self.config.seed is not None
            else None
        )

        train_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.config.batch_size_per_gpu,
            shuffle=(train_sampler is None),
            num_workers=self.config.num_worker_per_gpu,
            sampler=train_sampler,
            collate_fn=None,
            pin_memory=True,
            worker_init_fn=init_fn,
            drop_last=True,
            persistent_workers=True,
        )
        return train_loader

    def build_val_loader(self):
        val_data = build_dataset(self.config.val_dataset)
        if comm.get_world_size() > 1:
            val_sampler = torch.utils.data.distributed.DistributedSampler(val_data)
        else:
            val_sampler = None
        val_loader = torch.utils.data.DataLoader(
                val_data,
                batch_size=self.config.batch_size_val_per_gpu,
                shuffle=False,
                num_workers=self.config.num_worker_per_gpu,
                pin_memory=True,
                sampler=val_sampler
        )
        return val_loader

    def build_optimizer(self):
        return build_optimizer(self.config.optimizer, self.model, self.config.param_dicts)


    def build_scheduler(self):
        if 'epochs' not in self.config.scheduler or self.config.scheduler.steps_per_epoch==0:
            steps_per_epoch = len(self.dataloader_train)
            self.config.scheduler.steps_per_epoch = steps_per_epoch
        if 'epochs' not in self.config.scheduler:
            self.config.scheduler.epochs = self.config.eval_epoch
        return build_schedulers(self.config.scheduler, self.optimizer)


    def train(self):
        with EventStorage() as self.storage, ExceptionWriter():
            # => before train
            self.before_train()
            self.logger.info(">>>>>>>>>>>>>>>> Start Training >>>>>>>>>>>>>>>>")
            for self.epoch in range(self.start_epoch, self.max_epoch):
                # => before epoch
                # TODO: optimize to iteration based
                if comm.get_world_size() > 1:
                    self.train_loader.sampler.set_epoch(self.epoch)
                self.model.train()
                self.data_iterator = enumerate(self.dataloader_train)
                self.before_epoch()
                # => run_epoch
                for (
                    self.comm_info["iter"],
                    self.comm_info["input_dict"],
                ) in self.data_iterator:
                    # => before_step
                    self.before_step()
                    # => run_step
                    self.run_step()
                    # => after_step
                    self.after_step()
                # => after epoch
                self.after_epoch()
            # => after train
            self.after_train()

    def run_step(self):
        input_dict = self.comm_info["input_dict"]
        for key in input_dict.keys():
            if isinstance(input_dict[key], torch.Tensor):
                input_dict[key] = input_dict[key].cuda(non_blocking=True)
        with torch.cuda.amp.autocast(enabled=self.config.enable_amp):
            output_dict = self.model(input_dict)
            loss = output_dict["loss"]
            # self.logger.info(f'loss {loss}')
        self.optimizer.zero_grad()
        if self.config.enable_amp:
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            if self.config.clip_grad is not None:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.config.clip_grad
                )
            self.scaler.step(self.optimizer)

            # When enable amp, optimizer.step call are skipped if the loss scaling factor is too large.
            # Fix torch warning scheduler step before optimizer step.
            scaler = self.scaler.get_scale()
            self.scaler.update()
            if scaler <= self.scaler.get_scale():
                self.scheduler.step()
        else:
            loss.backward()
            if self.config.clip_grad is not None:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.config.clip_grad
                )
            self.optimizer.step()
            self.scheduler.step()
        if self.config.empty_cache:
            torch.cuda.empty_cache()
        self.comm_info["model_output_dict"] = output_dict





