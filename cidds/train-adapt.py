#!/usr/bin/env python3 -u
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Train a new model on one or across multiple GPUs.
"""

import logging
import math
import os
import random
import sys

import numpy as np
import torch

from fairseq import (
    checkpoint_utils,
    distributed_utils,
    options,
    quantization_utils,
    tasks,
    utils,
)
from fairseq.checkpoint_utils import torch_persistent_save
from fairseq.data import iterators, data_utils
from fairseq.file_io import PathManager
from fairseq.logging import meters, metrics, progress_bar
from fairseq.trainer import Trainer
from fairseq.model_parallel.megatron_trainer import MegatronTrainer
from itertools import zip_longest
from torch.serialization import default_restore_location

logging.basicConfig(
    format='%(asctime)s | %(levelname)s | %(name)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    level=logging.INFO,
    stream=sys.stdout,
)
logger = logging.getLogger('fairseq_cli.train')


class UDATrainer(Trainer):
    def __init__(self, args, task, model, criterion, quantizer=None):
        super().__init__(args, task, model, criterion, quantizer)

    def get_mono_source_iterator(self, epoch, shard_batch_itr=True,
    ):
        """Return an EpochBatchIterator over the training set for a given epoch."""
        return self.get_mono_iterator("unlabel_source", epoch, shard_batch_itr)

    def get_mono_target_iterator(self, epoch, shard_batch_itr=True,
    ):
        """Return an EpochBatchIterator over the training set for a given epoch."""
        return self.get_mono_iterator("unlabel_target", epoch, shard_batch_itr)

    def get_mono_iterator(self, split, epoch, shard_batch_itr=True,
    ):
        """Return an EpochBatchIterator over the training set for a given epoch."""
        return self.task.get_batch_iterator(
            dataset=self.task.dataset(split),
            max_tokens=self.args.max_tokens,
            max_sentences=self.args.max_sentences,
            max_positions=utils.resolve_max_positions(
                self.task.max_positions(),
                self.model.max_positions(),
                self.args.max_tokens,
            ),
            ignore_invalid_inputs=True,
            required_batch_size_multiple=self.args.required_batch_size_multiple,
            seed=self.args.seed,
            num_shards=self.data_parallel_world_size if shard_batch_itr else 1,
            shard_id=self.data_parallel_rank if shard_batch_itr else 0,
            num_workers=self.args.num_workers,
            epoch=epoch
        )


def load_pretrained_model(model, pretrained_path):
    bexists = PathManager.isfile(pretrained_path)
    if bexists:
        with PathManager.open(pretrained_path, "rb") as f:
            state = torch.load(
                f, map_location=lambda s, l: default_restore_location(s, "cpu")
            )
        model.load_state_dict(
            state["model"], strict=True
        )
    else:
        logger.warning("Cannot file checkpoint {}".format(pretrained_path))
        return model
    return model

def main(args, init_distributed=False):
    utils.import_user_module(args)

    assert args.max_tokens is not None or args.max_sentences is not None, \
        'Must specify batch size either with --max-tokens or --max-sentences'
    metrics.reset()

    # Initialize CUDA and distributed training
    if torch.cuda.is_available() and not args.cpu and not getattr(args, 'tpu', False):
        torch.cuda.set_device(args.device_id)
    np.random.seed(args.seed)
    utils.set_torch_seed(args.seed)
    if init_distributed:
        args.distributed_rank = distributed_utils.distributed_init(args)

    if distributed_utils.is_master(args):
        checkpoint_utils.verify_checkpoint_directory(args.save_dir)

    # Print args
    logger.info(args)

    # Setup task, e.g., translation, language modeling, etc.
    task = tasks.setup_task(args)

    # Load valid dataset (we load training data below, based on the latest checkpoint)
    for valid_sub_split in args.valid_subset.split(','):
        task.load_dataset(valid_sub_split, combine=False, epoch=1)

    # Build model and criterion
    model = task.build_model(args)

    if hasattr(args, 'pretrained_nmt') and args.pretrained_nmt:
        model_checkpoint = args.pretrained_nmt
        logger.info("Load NMT models from {}".format(model_checkpoint))
        model.nmt = load_pretrained_model(model.nmt, model_checkpoint)

    criterion = task.build_criterion(args)
    logger.info(model)
    logger.info('model {}, criterion {}'.format(args.arch, criterion.__class__.__name__))
    logger.info('num. model params: {} (num. trained: {})'.format(
        sum(p.numel() for p in model.parameters()),
        sum(p.numel() for p in model.parameters() if p.requires_grad),
    ))

    # (optionally) Configure quantization
    if args.quantization_config_path is not None:
        quantizer = quantization_utils.Quantizer(
            config_path=args.quantization_config_path,
            max_epoch=args.max_epoch,
            max_update=args.max_update,
        )
    else:
        quantizer = None

    # Build trainer
    if args.model_parallel_size == 1:
        trainer = UDATrainer(args, task, model, criterion, quantizer)
    else:
        trainer = MegatronTrainer(args, task, model, criterion)

    logger.info('training on {} devices (GPUs/TPUs)'.format(args.distributed_world_size))
    logger.info('max tokens per GPU = {} and max sentences per GPU = {}'.format(
        args.max_tokens,
        args.max_sentences,
    ))

    # Load the latest checkpoint if one is available and restore the
    # corresponding train iterator
    extra_state, epoch_itr = checkpoint_utils.load_checkpoint(args, trainer)
    if args.tpu:
        import torch_xla.core.xla_model as xm
        xm.rendezvous('load_checkpoint')  # wait for all workers
        xm.mark_step()

    # Train until the learning rate gets too small
    max_epoch = args.max_epoch or math.inf
    lr = trainer.get_lr()
    train_meter = meters.StopwatchMeter()
    train_meter.start()

    mono_src_itr = trainer.get_mono_source_iterator(epoch = epoch_itr.next_epoch_idx)
    mono_tgt_itr = trainer.get_mono_target_iterator(epoch = epoch_itr.next_epoch_idx)
    while (
        lr > args.min_lr
        and epoch_itr.next_epoch_idx <= max_epoch
    ):
        # train for one epoch
        valid_losses, should_stop = train(args, trainer, task, epoch_itr, mono_src_itr, mono_tgt_itr)
        if should_stop:
            break

        # only use first validation loss to update the learning rate
        lr = trainer.lr_step(epoch_itr.epoch, valid_losses[0])

        epoch_itr = trainer.get_train_iterator(
            epoch_itr.next_epoch_idx,
            # sharded data: get train iterator for next epoch
            load_dataset=(os.pathsep in getattr(args, 'data', '')),
        )
        mono_src_itr = trainer.get_mono_source_iterator(epoch=epoch_itr.next_epoch_idx)
        mono_tgt_itr = trainer.get_mono_target_iterator(epoch=epoch_itr.next_epoch_idx)
    train_meter.stop()
    logger.info('done training in {:.1f} seconds'.format(train_meter.sum))

    logger.info('save MT')
    save_nmt_models(args, model)

def save_nmt_models(args, models):
    # load best model
    best_ckpt = f"{args.save_dir}/checkpoint_best.pt"
    last_ckpt = f"{args.save_dir}/checkpoint_last.pt"

    if not PathManager.exists(best_ckpt):
        logger.info("Best checkpoint does not exist. Use last checkpoint.")
        best_ckpt = f"{args.save_dir}/checkpoint_last.pt"
    with PathManager.open(best_ckpt, "rb") as f:
        state = torch.load(
            f, map_location=lambda s, l: default_restore_location(s, "cpu")
        )
    models.load_state_dict(
        state["model"], strict=True
    )

    nmt_state_dict = models.nmt.state_dict()

    state_dict = {
        "model": nmt_state_dict,
    }
    state_dict = utils.move_to_cpu(state_dict)

    output_ckpt = f"{args.save_dir}/model.pt"
    logger.info(f"Save best model to {output_ckpt}")
    with PathManager.open(output_ckpt, "wb") as f:
        torch_persistent_save(state_dict, f)


def should_stop_early(args, valid_loss):
    # skip check if no validation was done in the current epoch
    if valid_loss is None:
        return False
    if args.patience <= 0:
        return False

    def is_better(a, b):
        return a > b if args.maximize_best_checkpoint_metric else a < b

    prev_best = getattr(should_stop_early, 'best', None)
    if prev_best is None or is_better(valid_loss, prev_best):
        should_stop_early.best = valid_loss
        should_stop_early.num_runs = 0
        return False
    else:
        should_stop_early.num_runs += 1
        if should_stop_early.num_runs >= args.patience:
            logger.info('early stop since valid performance hasn\'t improved for last {} runs'.format(args.patience))
            return True
        else:
            return False


def tpu_data_loader(args, itr):
    import torch_xla.core.xla_model as xm
    import torch_xla.distributed.parallel_loader as pl
    xm.rendezvous('tpu_data_loader')  # wait for all workers
    xm.mark_step()
    device = utils.get_tpu_device(args)
    return iterators.CountingIterator(
        pl.ParallelLoader(itr, [device]).per_device_loader(device),
        start=getattr(itr, 'n', 0),
        total=len(itr),
    )

def collate(parallel_samples, source_samples, target_samples, src_dict, tgt_dict):
    collated_samples = {
        'parallel_id': parallel_samples['id'],
        'source_id': source_samples['id'],
        'target_id': target_samples['id'],
        'parallel_nsentences': parallel_samples['nsentences'],
        'source_nsentences': source_samples['nsentences'],
        'target_nsentences': target_samples['nsentences'],
        'parallel_ntokens': parallel_samples['ntokens'],
        'source_ntokens': source_samples['ntokens'],
        'target_ntokens': target_samples['ntokens'],
        'net_input':{
            'src_tokens': parallel_samples['net_input']['src_tokens'],
            'src_lengths':parallel_samples['net_input']['src_lengths'],
            'prev_output_tokens': parallel_samples['net_input']['prev_output_tokens'],
            'mono_src_tokens': source_samples['net_input']['src_tokens'],
            "mono_src_lengths": source_samples['net_input']['src_lengths'],
            'mono_tgt_tokens': target_samples['net_input']['src_tokens'],
            'mono_prev_output_tokens':
                data_utils.collate_tokens( target_samples['net_input']['src_tokens'],
                tgt_dict.pad(), tgt_dict.eos(), move_eos_to_beginning=True),
            'dummy_src_tokens': torch.ones_like(target_samples['target']).unsqueeze(1) * tgt_dict.eos_index,
            'dummy_src_lengths': torch.ones_like(target_samples['target']),
        },
        'parallel_target': parallel_samples['target'],
        'source_target': source_samples['target'],
        'target_target': target_samples['target'],
    }

    return collated_samples

@metrics.aggregate('train')
def train(args, trainer, task, epoch_itr, mono_src_itr, mono_tgt_itr):
    """Train the model for one epoch and return validation losses."""
    # Initialize data iterator
    nmt_itr = epoch_itr.next_epoch_itr(
        fix_batches_to_gpus=args.fix_batches_to_gpus,
        shuffle=(epoch_itr.next_epoch_idx > args.curriculum),
    )
    update_freq = (
        args.update_freq[epoch_itr.epoch - 1]
        if epoch_itr.epoch <= len(args.update_freq)
        else args.update_freq[-1]
    )
    nmt_itr = iterators.GroupedIterator(nmt_itr, update_freq)
    src_itr = mono_src_itr.next_epoch_itr(
        fix_batches_to_gpus=args.fix_batches_to_gpus,
        shuffle=True,
    )
    src_itr = iterators.GroupedIterator(src_itr, update_freq)
    tgt_itr = mono_tgt_itr.next_epoch_itr(
        fix_batches_to_gpus=args.fix_batches_to_gpus,
        shuffle=True,
    )
    tgt_itr = iterators.GroupedIterator(tgt_itr, update_freq)

    nmt_progress = progress_bar.progress_bar(
        nmt_itr,
        log_format=args.log_format,
        log_interval=args.log_interval,
        epoch=epoch_itr.epoch,
        tensorboard_logdir=(
            args.tensorboard_logdir if distributed_utils.is_master(args) else None
        ),
        default_log_format=('tqdm' if not args.no_progress_bar else 'simple'),
    )

    src_progress = progress_bar.progress_bar(
      src_itr,
      log_format=args.log_format,
      log_interval=args.log_interval,
      epoch=epoch_itr.epoch,
      tensorboard_logdir=(
        args.tensorboard_logdir if distributed_utils.is_master(args) else None
      ),
      default_log_format=('tqdm' if not args.no_progress_bar else 'simple'),
    )

    tgt_progress = progress_bar.progress_bar(
        tgt_itr,
        log_format=args.log_format,
        log_interval=args.log_interval,
        epoch=epoch_itr.epoch,
        tensorboard_logdir=(
            args.tensorboard_logdir if distributed_utils.is_master(args) else None
        ),
        default_log_format=('tqdm' if not args.no_progress_bar else 'simple'),
    )

    trainer.begin_epoch(epoch_itr.epoch)
    src_dict = task.src_dict
    tgt_dict = task.tgt_dict
    valid_subsets = args.valid_subset.split(',')
    should_stop = False
    num_updates = trainer.get_num_updates()
    for nmt_samples, src_samples, tgt_samples in zip_longest(nmt_progress, src_progress, tgt_progress):
        if nmt_samples is None or src_samples is None or tgt_samples is None:
          logger.info("[GPU {}] Drain out of data".format(args.device_id))
          break
        num_parallel = nmt_samples[0]['nsentences']
        num_source = src_samples[0]['nsentences']
        num_target = tgt_samples[0]['nsentences']
        if num_parallel != num_source or num_source != num_target or num_parallel == 0 or num_source == 0 or num_target == 0:
            logger.info("[GPU {}] Drain out of data".format(args.device_id))
            break
        with metrics.aggregate('train_inner'):
            log_output = trainer.train_step([collate(nmt_samples[0], src_samples[0], tgt_samples[0], src_dict, tgt_dict)])
            if log_output is None:  # OOM, overflow, ...
                continue

        # log mid-epoch stats
        num_updates = trainer.get_num_updates()
        if num_updates % args.log_interval == 0:
            stats = get_training_stats(metrics.get_smoothed_values('train_inner'))
            nmt_progress.log(stats, tag='train_inner', step=num_updates)

            # reset mid-epoch stats after each log interval
            # the end-of-epoch stats will still be preserved
            metrics.reset_meters('train_inner')

    end_of_epoch = True
    valid_losses, should_stop = validate_and_save(
        args, trainer, task, epoch_itr, valid_subsets, end_of_epoch
    )

    # log end-of-epoch stats
    try:
        stats = get_training_stats(metrics.get_smoothed_values('train'))
        nmt_progress.print(stats, tag='train', step=num_updates)
    except:
        logger.warning("Not stat found")


    # reset epoch-level meters
    metrics.reset_meters('train')
    return valid_losses, should_stop


def validate_and_save(args, trainer, task, epoch_itr, valid_subsets, end_of_epoch):
    num_updates = trainer.get_num_updates()
    do_save = (
        (
            args.save_interval_updates > 0
            and num_updates > 0
            and num_updates % args.save_interval_updates == 0
        )
        or (end_of_epoch and epoch_itr.epoch % args.save_interval == 0)
    )
    do_validate = (
        (
            (not end_of_epoch and do_save)  # validate during mid-epoch saves
            or (end_of_epoch and epoch_itr.epoch % args.validate_interval == 0)
        )
        and not args.disable_validation
    )

    # Validate
    valid_losses = [None]
    if do_validate:
        valid_losses = validate(args, trainer, task, epoch_itr, valid_subsets)

    # Stopping conditions
    max_update = args.max_update or math.inf
    should_stop = (
        should_stop_early(args, valid_losses[0])
        or trainer.get_num_updates() >= max_update
    )

    # Save checkpoint
    if do_save or should_stop:
        checkpoint_utils.save_checkpoint(args, trainer, epoch_itr, valid_losses[0])

    return valid_losses, should_stop

def save_model_state_dict(model_state_dict, filename):
    logger.info("Save model to {}".format(filename))
    state_dict = {
        "model": model_state_dict,
        "args": {}
    }
    state_dict = utils.move_to_cpu(state_dict)
    with PathManager.open(filename, "wb") as f:
        torch_persistent_save(state_dict, f)

def get_training_stats(stats):
    stats['wall'] = round(metrics.get_meter('default', 'wall').elapsed_time, 0)
    return stats


def validate(args, trainer, task, epoch_itr, subsets):
    """Evaluate the model on the validation set(s) and return the losses."""

    if args.fixed_validation_seed is not None:
        # set fixed seed for every validation
        utils.set_torch_seed(args.fixed_validation_seed)

    valid_losses = []
    for subset in subsets:
        # Initialize data iterator
        itr = trainer.get_valid_iterator(subset).next_epoch_itr(shuffle=False)
        if getattr(args, 'tpu', False):
            itr = tpu_data_loader(args, itr)
        progress = progress_bar.progress_bar(
            itr,
            log_format=args.log_format,
            log_interval=args.log_interval,
            epoch=epoch_itr.epoch,
            prefix=f"valid on '{subset}' subset",
            tensorboard_logdir=(
                args.tensorboard_logdir if distributed_utils.is_master(args) else None
            ),
            default_log_format=('tqdm' if not args.no_progress_bar else 'simple'),
        )

        # create a new root metrics aggregator so validation metrics
        # don't pollute other aggregators (e.g., train meters)
        with metrics.aggregate(new_root=True) as agg:
            for sample in progress:
                trainer.valid_step(sample)

        # log validation stats
        stats = get_valid_stats(args, trainer, agg.get_smoothed_values())
        progress.print(stats, tag=subset, step=trainer.get_num_updates())

        valid_losses.append(stats[args.best_checkpoint_metric])
    return valid_losses


def get_valid_stats(args, trainer, stats):
    stats['num_updates'] = trainer.get_num_updates()
    if hasattr(checkpoint_utils.save_checkpoint, 'best'):
        key = 'best_{0}'.format(args.best_checkpoint_metric)
        best_function = max if args.maximize_best_checkpoint_metric else min
        stats[key] = best_function(
            checkpoint_utils.save_checkpoint.best,
            stats[args.best_checkpoint_metric],
        )
    return stats


def distributed_main(i, args, start_rank=0):
    args.device_id = i
    if args.distributed_rank is None:  # torch.multiprocessing.spawn
        args.distributed_rank = start_rank + i
    main(args, init_distributed=True)


def fairseq_train(modify_parser=None):
    parser = options.get_training_parser()
    args = options.parse_args_and_arch(parser, modify_parser=modify_parser)
    if args.distributed_init_method is None:
        distributed_utils.infer_init_method(args)

    if args.distributed_init_method is not None:
        # distributed training
        if torch.cuda.device_count() > 1 and not args.distributed_no_spawn:
            start_rank = args.distributed_rank
            args.distributed_rank = None  # assign automatically
            torch.multiprocessing.spawn(
                fn=distributed_main,
                args=(args, start_rank),
                nprocs=torch.cuda.device_count(),
            )
        else:
            distributed_main(args.device_id, args)
    elif args.distributed_world_size > 1:
        if not getattr(args, 'tpu', False):
            # fallback for single node with multiple GPUs
            assert args.distributed_world_size <= torch.cuda.device_count()
            port = random.randint(10000, 20000)
            args.distributed_init_method = 'tcp://localhost:{port}'.format(port=port)
            args.distributed_rank = None  # set based on device id
            torch.multiprocessing.spawn(
                fn=distributed_main,
                args=(args, ),
                nprocs=args.distributed_world_size,
            )
        else:
            import torch_xla.distributed.xla_multiprocessing as xmp
            torch.multiprocessing.set_sharing_strategy('file_system')
            xmp.spawn(
                fn=distributed_main,
                args=(args, ),
                nprocs=8,  # use all 8 TPU cores
            )
    else:
        # single GPU training
        main(args)

if __name__ == '__main__':
    fairseq_train()
