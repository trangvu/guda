# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os

import numpy as np
import torch
from torch.serialization import default_restore_location

from al.util import load_model_from_checkpoint
from fairseq.data import (
    data_utils,
    Dictionary,
    IdDataset,
    MaskTokensDataset,
    NestedDictionaryDataset,
    NumelDataset,
    NumSamplesDataset,
    PadDataset,
    PrependTokenDataset,
    SortDataset,
    TokenBlockDataset,
)
from fairseq.data.dictionary import PrebuiltDictionary
from fairseq.data.shorten_dataset import maybe_shorten_dataset
from fairseq.file_io import PathManager
from fairseq.tasks import FairseqTask, register_task
from fairseq.data.encoders.utils import get_whole_word_mask
from fairseq import utils
from fairseq.tasks.translation import load_langpair_dataset

logger = logging.getLogger(__name__)


@register_task('contrastive_nmt')
class ContrastiveNMTTask(FairseqTask):
    """Task for training masked language models (e.g., BERT, RoBERTa)."""

    @staticmethod
    def add_args(parser):
        """Add task-specific arguments to the parser."""
        parser.add_argument('data', help='colon separated path to data directories list, \
                            will be iterated upon during epochs in round-robin manner')
        parser.add_argument('--sample-break-mode', default='complete',
                            choices=['none', 'complete', 'complete_doc', 'eos'],
                            help='If omitted or "none", fills each sample with tokens-per-sample '
                                 'tokens. If set to "complete", splits samples only at the end '
                                 'of sentence, but may include multiple sentences per sample. '
                                 '"complete_doc" is similar but respects doc boundaries. '
                                 'If set to "eos", includes only one sentence per sample.')
        parser.add_argument('--tokens-per-sample', default=512, type=int,
                            help='max number of total tokens over all segments '
                                 'per sample for BERT dataset')
        parser.add_argument('--mask-prob', default=0.15, type=float,
                            help='probability of replacing a token with mask')
        parser.add_argument('--leave-unmasked-prob', default=0.1, type=float,
                            help='probability that a masked token is unmasked')
        parser.add_argument('--random-token-prob', default=0.1, type=float,
                            help='probability of replacing a token with a random token')
        parser.add_argument('--freq-weighted-replacement', default=False, action='store_true',
                            help='sample random replacement words based on word frequencies')
        parser.add_argument('--mask-whole-words', default=False, action='store_true',
                            help='mask whole words; you may also want to set --bpe')
        parser.add_argument('--shorten-method', default='none',
                            choices=['none', 'truncate', 'random_crop'],
                            help='if not none, shorten sequences that exceed --tokens-per-sample')
        parser.add_argument('--shorten-data-split-whitelist', default='',
                            help='comma-separated list of dataset splits to apply shortening to, '
                                 'e.g., "train,valid" (default: all dataset splits)')
        parser.add_argument('--offset-positions-by-padding', default=True, action='store_false',
                            help='Offset positions by padding')
        parser.add_argument('--temperature', default=1, type=float, help="Temperature to compute cross entropy loss")
        parser.add_argument('--config-file', help="Config file")
        parser.add_argument('--source-model', help="Source model file")
        parser.add_argument('--source-vocab', help="Path to source vocab")
        parser.add_argument('--target-model', help="Path to target model")
        parser.add_argument('--target-vocab', help="Path to target vocab")
        parser.add_argument('--source-lang', help="Source language")
        parser.add_argument('--target-lang', help="Target language")

    def __init__(self, args, src_dict, tgt_dict):
        super().__init__(args)
        self.src_dict = src_dict
        self.tgt_dict = tgt_dict
        self.seed = args.seed

        # add mask token
        self.src_mask_idx = src_dict.index('<mask>')
        self.tgt_mask_idx = tgt_dict.index('<mask>')

    @classmethod
    def setup_task(cls, args, **kwargs):
        src_paths = utils.split_paths(args.source_model)
        tgt_paths = utils.split_paths(args.target_model)
        assert len(src_paths) > 0
        assert len(tgt_paths) > 0
        src_dict = PrebuiltDictionary.load(os.path.join(src_paths[0], 'dict.txt'))
        src_dict.init_special_tokens()
        logger.info('Source dictionary: {} types'.format(len(src_dict)))

        tgt_dict = PrebuiltDictionary.load(os.path.join(tgt_paths[0], 'dict.txt'))
        tgt_dict.init_special_tokens()
        logger.info('Target dictionary: {} types'.format(len(tgt_dict)))
        return cls(args, src_dict, tgt_dict)

    @classmethod
    def load_dictionary(self, filename):
        src_dict = PrebuiltDictionary.load(filename)
        src_dict.init_special_tokens()
        return src_dict

    def load_dataset(self, split, epoch=1, combine=False, **kwargs):
        """Load a given dataset split.

        Args:
            split (str): name of the split (e.g., train, valid, test)
        """
        paths = utils.split_paths(self.args.data)
        assert len(paths) > 0
        data_path = paths[(epoch - 1) % len(paths)]
        src, tgt = self.args.source_lang, self.args.target_lang
        src_filename = '{}.{}-{}.{}'.format(split, src, tgt, src)
        tgt_filename = '{}.{}-{}.{}'.format(split, src, tgt, tgt)
        src_split_path = os.path.join(data_path, src_filename)
        tgt_split_path = os.path.join(data_path, tgt_filename)

        src_dataset = data_utils.load_indexed_dataset(
            src_split_path,
            self.source_dictionary,
            self.args.dataset_impl,
            combine=combine,
        )
        if src_dataset is None:
            raise FileNotFoundError('Dataset not found: {} ({})'.format(split, src_split_path))

        tgt_dataset = data_utils.load_indexed_dataset(
            tgt_split_path,
            self.target_dictionary,
            self.args.dataset_impl,
            combine=combine,
        )
        if tgt_dataset is None:
            raise FileNotFoundError('Dataset not found: {} ({})'.format(split, tgt_split_path))

        # prepend beginning-of-sentence token (<s>, equiv. to [CLS] in BERT)
        src_dataset = PrependTokenDataset(src_dataset, self.source_dictionary.bos())
        tgt_dataset = PrependTokenDataset(tgt_dataset, self.target_dictionary.bos())

        with data_utils.numpy_seed(self.args.seed + epoch):
            shuffle = np.random.permutation(len(src_dataset))

        self.datasets[split] = SortDataset(
            NestedDictionaryDataset(
                {
                    'id': IdDataset(),
                    'net_input': {
                        'src_tokens': PadDataset(
                            src_dataset,
                            pad_idx=self.source_dictionary.pad(),
                            left_pad=False,
                        ),
                        'src_lengths': NumelDataset(src_dataset, reduce=False),
                        'tgt_tokens': PadDataset(
                            tgt_dataset,
                            pad_idx=self.target_dictionary.pad(),
                            left_pad=False,
                        ),
                        'tgt_lengths': NumelDataset(tgt_dataset, reduce=False),
                    },
                    'nsentences': NumSamplesDataset(),
                    'ntokens': NumelDataset(src_dataset, reduce=True),
                },
                sizes=[src_dataset.sizes],
            ),
            sort_order=[
                shuffle,
                src_dataset.sizes,
            ],
        )

    def build_dataset_for_inference(self, src_tokens, src_lengths, sort=True):
        src_dataset = PadDataset(
            TokenBlockDataset(
                src_tokens,
                src_lengths,
                self.args.tokens_per_sample - 1,  # one less for <s>
                pad=self.source_dictionary.pad(),
                eos=self.source_dictionary.eos(),
                break_mode='eos',
            ),
            pad_idx=self.source_dictionary.pad(),
            left_pad=False,
        )
        src_dataset = PrependTokenDataset(src_dataset, self.source_dictionary.bos())
        src_dataset = NestedDictionaryDataset(
            {
                'id': IdDataset(),
                'net_input': {
                    'src_tokens': src_dataset,
                    'src_lengths': NumelDataset(src_dataset, reduce=False),
                },
            },
            sizes=src_lengths,
        )
        if sort:
            src_dataset = SortDataset(src_dataset, sort_order=[src_lengths])
        return src_dataset

    @property
    def source_dictionary(self):
        return self.src_dict

    @property
    def target_dictionary(self):
        return self.tgt_dict
