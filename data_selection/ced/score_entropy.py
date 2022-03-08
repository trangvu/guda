#!/usr/bin/env python3 -u
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Evaluate the perplexity of a trained language model.
"""

import logging
import math
import os

import torch

from fairseq import checkpoint_utils, options, tasks, utils
from fairseq.data import LMContextWindowDataset, data_utils, TokenBlockDataset, MonolingualDataset
from fairseq.data.shorten_dataset import maybe_shorten_dataset
from fairseq.logging import progress_bar
from fairseq.logging.meters import StopwatchMeter, TimeMeter
from fairseq.sequence_scorer import SequenceScorer
from fairseq.options import add_distributed_training_args
from fairseq import distributed_utils


logging.basicConfig(
    format='%(asctime)s | %(levelname)s | %(name)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def load_dataset(task, split, epoch=1, combine=False, **kwargs):
    """Load a given dataset split.

    Args:
        split (str): name of the split (e.g., train, valid, test)
    """
    paths = utils.split_paths(task.args.data)
    assert len(paths) > 0

    data_path = paths[(epoch - 1) % len(paths)]
    split_path = os.path.join(data_path, split)

    dataset = data_utils.load_indexed_dataset(
        split_path, task.dictionary, task.args.dataset_impl, combine=combine
    )
    if dataset is None:
        raise FileNotFoundError(
            "Dataset not found: {} ({})".format(split, split_path)
        )


    dataset = MonolingualDataset(
        dataset,
        dataset.sizes,
        task.dictionary,
        task.output_dictionary,
        add_eos_for_other_targets=True,
        shuffle=False,
        targets=task.targets,
        add_bos_token=task.args.add_bos_token,
    )
    return dataset


def main(parsed_args, **unused_kwargs):
    assert parsed_args.path is not None, '--path required for evaluation!'

    if torch.cuda.is_available() and not parsed_args.cpu:
        torch.cuda.set_device(parsed_args.device_id)

    utils.import_user_module(parsed_args)

    logger.info(parsed_args)

    use_cuda = torch.cuda.is_available() and not parsed_args.cpu

    task = tasks.setup_task(parsed_args)

    # Load ensemble
    logger.info('loading model(s) from {}'.format(parsed_args.path))
    models, args = checkpoint_utils.load_model_ensemble(
        parsed_args.path.split(os.pathsep),
        arg_overrides=eval(parsed_args.model_overrides),
        task=task,
        suffix=getattr(parsed_args, "checkpoint_suffix", ""),
    )

    for arg in vars(parsed_args).keys():
        if arg not in {
            'self_target', 'future_target', 'past_target', 'tokens_per_sample',
            'output_size_dictionary', 'add_bos_token',
        }:
            setattr(args, arg, getattr(parsed_args, arg))

    # reduce tokens per sample by the required context window size
    task = tasks.setup_task(args)

    # Load dataset splits
    # dataset = load_dataset(task, args.gen_subset)
    task.load_dataset(args.gen_subset)
    dataset = task.dataset(args.gen_subset)
    logger.info('{} {} {} examples'.format(args.data, args.gen_subset, len(dataset)))
    # Optimize ensemble for generation and set the source and dest dicts on the model (required by scorer)
    for model in models:
        model.make_generation_fast_()
        if args.fp16:
            model.half()
        if use_cuda:
            model.cuda()

    assert len(models) > 0

    logger.info('num. model params: {}'.format(sum(p.numel() for p in models[0].parameters())))

    itr = task.get_batch_iterator(
        dataset=dataset,
        max_tokens=args.max_tokens or 36000,
        max_sentences=args.max_sentences,
        max_positions=utils.resolve_max_positions(*[
            model.max_positions() for model in models
        ]),
        ignore_invalid_inputs=False,
        num_shards=args.num_shards,
        shard_id=args.shard_id,
        num_workers=args.num_workers,
    ).next_epoch_itr(shuffle=False)
    progress = progress_bar.progress_bar(
        itr,
        log_format=args.log_format,
        log_interval=args.log_interval,
        default_log_format=('tqdm' if not args.no_progress_bar else 'none'),
    )

    gen_timer = StopwatchMeter()
    scorer = SequenceScorer(task.target_dictionary, args.softmax_batch)

    score_sum = 0.
    count = 0

    if args.remove_bpe is not None:
        if args.remove_bpe == 'sentencepiece':
            raise NotImplementedError
        else:
            bpe_cont = args.remove_bpe.rstrip()
            bpe_toks = {
                i
                for i in range(len(task.source_dictionary))
                if task.source_dictionary[i].endswith(bpe_cont)
            }
        bpe_len = len(bpe_cont)
    else:
        bpe_toks = None
        bpe_len = 0
    src_dict = task.source_dictionary
    wps_meter = TimeMeter()

    dirname = os.path.dirname(args.score_output)
    os.makedirs(dirname, exist_ok=True)
    with open(args.score_output, 'w') as fout:
        for sample in progress:
            if 'net_input' not in sample:
                continue

            sample = utils.move_to_cuda(sample) if use_cuda else sample

            gen_timer.start()
            hypos = scorer.generate(models, sample)
            gen_timer.stop(sample['ntokens'])

            for i, hypos_i in enumerate(hypos):
                hypo = hypos_i[0]
                sample_id = sample['id'][i]

                tokens = hypo['tokens']
                tgt_len = tokens.numel()
                pos_scores = hypo['positional_scores'].float()

                if args.add_bos_token:
                    assert hypo['tokens'][0].item() == task.target_dictionary.bos()
                    tokens = tokens[1:]
                    pos_scores = pos_scores[1:]

                skipped_toks = 0
                if bpe_toks is not None:
                    for i in range(tgt_len - 1):
                        if tokens[i].item() in bpe_toks:
                            skipped_toks += 1
                            pos_scores[i + 1] += pos_scores[i]
                            pos_scores[i] = 0

                inf_scores = pos_scores.eq(float('inf')) | pos_scores.eq(float('-inf'))
                if inf_scores.any():
                    logger.info(
                        'skipping tokens with inf scores:',
                        task.target_dictionary.string(tokens[inf_scores.nonzero()])
                    )
                    pos_scores = pos_scores[(~inf_scores).nonzero()]

                score = pos_scores.sum().cpu()
                cnt = pos_scores.numel() - skipped_toks
                nll_loss = -score / cnt / math.log(2)
                # write score
                src_tokens = utils.strip_pad(sample['net_input']['src_tokens'][i, :], src_dict.pad())
                src_str = src_dict.string(src_tokens, args.remove_bpe)
                fout.write("{}\t{}\t{}\n".format(sample_id, nll_loss, src_str))
                score_sum += score
                count += cnt

        wps_meter.update(sample['ntokens'])
        progress.log({'wps': round(wps_meter.avg)})

    avg_nll_loss = -score_sum / count / math.log(2)  # convert to base 2
    logger.info('Evaluated {} tokens in {:.1f}s ({:.2f} tokens/s)'.format(
        gen_timer.n, gen_timer.sum, 1. / gen_timer.avg
    ))
    logger.info('Loss (base 2): {:.4f}, Perplexity: {:.2f}'.format(
        avg_nll_loss, 2**avg_nll_loss
    ))


def cli_main():
    parser = options.get_eval_lm_parser()
    group = parser.add_argument_group("Score entropy")
    group.add_argument('--score-output', required=True, help='Output score file')
    args = options.parse_args_and_arch(parser)
    distributed_utils.call_main(args, main)


if __name__ == '__main__':
    cli_main()
