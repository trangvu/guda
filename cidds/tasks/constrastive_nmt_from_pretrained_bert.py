'''
Created by trangvu on 22/01/21
'''
import itertools
import logging
import os

import torch

from fairseq import utils, options
from fairseq.data import NestedDictionaryDataset, IdDataset, LanguagePairDataset, data_utils, indexed_dataset, \
    AppendTokenDataset, PrependTokenDataset, ConcatDataset, TruncateDataset, StripTokenDataset, Dictionary, \
    OffsetTokensDataset, SortDataset
from fairseq.tasks import register_task
from fairseq.tasks.translation import TranslationTask

from ..data.pretrained_dictionary import PretrainedDictionary
import numpy as np

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


@register_task('contrastive_nmt_from_pretrained_bert_task')
class ContrastiveNMTFromPretrainedBertTask(TranslationTask):
    """Task for training masked language models (e.g., BERT, RoBERTa)."""

    def __init__(self, args, src_dict, tgt_dict, label_dict=None):
        super().__init__(args, src_dict, tgt_dict)
        self.label_dict = label_dict

    @staticmethod
    def add_args(parser):
      """Add task-specific arguments to the parser."""
      TranslationTask.add_args(parser)
      parser.add_argument('--temperature', default=1, type=float, help="Temperature to compute cross entropy loss")
      parser.add_argument('--precluster', action="store_true", help="Use precluster in constrastive loss")
      parser.add_argument('--labels', help="Path to cluster labels. Only use when precluster is true")
      parser.add_argument('--no_shuffle', action="store_true", help="No shuffle data")

    @classmethod
    def load_dictionary(cls, filename, source=True):
        """Load the masked LM dictionary from the filename

        Args:
            filename (str): the filename
        """
        if source:
            return PretrainedDictionary.load(filename)
        else:
            dictionary = Dictionary.load(filename)
            dictionary.add_symbol('<mask>')
            return dictionary

    @classmethod
    def setup_task(cls, args, **kwargs):
        args.left_pad_source = options.eval_bool(args.left_pad_source)
        args.left_pad_target = options.eval_bool(args.left_pad_target)

        paths = utils.split_paths(args.data)
        assert len(paths) > 0
        # find language pair automatically
        if args.source_lang is None or args.target_lang is None:
            args.source_lang, args.target_lang = data_utils.infer_language_pair(paths[0])
        if args.source_lang is None or args.target_lang is None:
            raise Exception('Could not infer language pair, please provide it explicitly')

        # load dictionaries
        train_data_path = "{}/{}".format(paths[0], "input0")
        src_dict = cls.load_dictionary(os.path.join(train_data_path, 'dict.{}.txt'.format(args.source_lang)))
        tgt_dict = cls.load_dictionary(os.path.join(train_data_path, 'dict.{}.txt'.format(args.target_lang)))
        assert src_dict.pad() == tgt_dict.pad()
        assert src_dict.eos() == tgt_dict.eos()
        assert src_dict.unk() == tgt_dict.unk()
        logger.info('[{}] dictionary: {} types'.format(args.source_lang, len(src_dict)))
        logger.info('[{}] dictionary: {} types'.format(args.target_lang, len(tgt_dict)))

        label_dict = None
        if args.precluster:
            data_label_path = os.path.join(args.data, args.labels, 'dict.txt')
            label_dict = cls.load_dictionary(
                data_label_path,
                source=False,
            )
            logger.info('[label] dictionary: {} types'.format(len(label_dict)))
        return cls(args, src_dict, tgt_dict, label_dict)

    def build_model(self, args):
        from fairseq import models
        model = models.build_model(args, self)

        model.register_adaptive_layer('adaptive')

        return model

    def load_dataset(self, split, epoch=1, combine=False, **kwargs):
        """Load a given dataset split.

        Args:
            split (str): name of the split (e.g., train, valid, test)
        """

        def get_path(type, split):
            return os.path.join(self.args.data, type, split)

        def make_dataset(type, dictionary):
            split_path = get_path(type, split)

            dataset = data_utils.load_indexed_dataset(
                split_path,
                dictionary,
                self.args.dataset_impl,
                combine=combine,
            )
            return dataset

        paths = utils.split_paths(self.args.data)
        assert len(paths) > 0
        data_path = paths[(epoch - 1) % len(paths)]

        # infer langcode
        src, tgt = self.args.source_lang, self.args.target_lang

        logger.info("Load trainning data")
        train_data_path = "{}/{}".format(data_path, "input0")
        dataset = load_langpair_dataset(
            train_data_path, split, src, self.src_dict, tgt, self.tgt_dict,
            combine=combine, dataset_impl=self.args.dataset_impl,
            upsample_primary=self.args.upsample_primary,
            left_pad_source=self.args.left_pad_source,
            left_pad_target=self.args.left_pad_target,
            max_source_positions=self.args.max_source_positions,
            max_target_positions=self.args.max_target_positions,
            load_alignments=self.args.load_alignments,
            truncate_source=self.args.truncate_source, )

        src_tokens = dataset.src
        with data_utils.numpy_seed(self.args.seed + epoch):
            shuffle = np.random.permutation(len(dataset))

        if self.args.precluster:
            logger.info("Load labels")
            label_dataset = make_dataset(self.args.labels, self.label_dict)
            if label_dataset is not None:
                dataset.update_label_dataset(
                    OffsetTokensDataset(
                        StripTokenDataset(
                            label_dataset,
                            id_to_strip=self.label_dict.eos(),
                        ),
                        offset=-self.label_dict.nspecial,
                    ),
                    self.label_dict, label_dataset.sizes
                )
            if not self.args.no_shuffle:
                dataset = SortDataset(
                    dataset,
                    # shuffle
                    sort_order=[shuffle],
                )
        logger.info("Loaded {0} with #samples: {1}".format(split, len(dataset)))
        self.datasets[split] = dataset


def load_langpair_dataset(
    data_path, split,
    src, src_dict,
    tgt, tgt_dict,
    combine, dataset_impl, upsample_primary,
    left_pad_source, left_pad_target, max_source_positions,
    max_target_positions, prepend_bos=False, load_alignments=False,
    truncate_source=False, append_source_id=False
):

    def split_exists(split, src, tgt, lang, data_path):
        filename = os.path.join(data_path, '{}.{}-{}.{}'.format(split, src, tgt, lang))
        return indexed_dataset.dataset_exists(filename, impl=dataset_impl)

    src_datasets = []
    tgt_datasets = []

    for k in itertools.count():
        split_k = split + (str(k) if k > 0 else '')

        # infer langcode
        if split_exists(split_k, src, tgt, src, data_path):
            prefix = os.path.join(data_path, '{}.{}-{}.'.format(split_k, src, tgt))
        elif split_exists(split_k, tgt, src, src, data_path):
            prefix = os.path.join(data_path, '{}.{}-{}.'.format(split_k, tgt, src))
        else:
            if k > 0:
                break
            else:
                raise FileNotFoundError('Dataset not found: {} ({})'.format(split, data_path))

        src_dataset = data_utils.load_indexed_dataset(prefix + src, src_dict, dataset_impl)
        if truncate_source:
            src_dataset = AppendTokenDataset(
                TruncateDataset(
                    StripTokenDataset(src_dataset, src_dict.eos()),
                    max_source_positions - 1,
                ),
                src_dict.eos(),
            )
        src_datasets.append(src_dataset)

        tgt_dataset = data_utils.load_indexed_dataset(prefix + tgt, tgt_dict, dataset_impl)
        if tgt_dataset is not None:
            tgt_datasets.append(tgt_dataset)

        logger.info('{} {} {}-{} {} examples'.format(
            data_path, split_k, src, tgt, len(src_datasets[-1])
        ))

        if not combine:
            break

    assert len(src_datasets) == len(tgt_datasets) or len(tgt_datasets) == 0

    if len(src_datasets) == 1:
        src_dataset = src_datasets[0]
        tgt_dataset = tgt_datasets[0] if len(tgt_datasets) > 0 else None
    else:
        sample_ratios = [1] * len(src_datasets)
        sample_ratios[0] = upsample_primary
        src_dataset = ConcatDataset(src_datasets, sample_ratios)
        if len(tgt_datasets) > 0:
            tgt_dataset = ConcatDataset(tgt_datasets, sample_ratios)
        else:
            tgt_dataset = None

    if prepend_bos:
        assert hasattr(src_dict, "bos_index") and hasattr(tgt_dict, "bos_index")
        src_dataset = PrependTokenDataset(src_dataset, src_dict.bos())
        if tgt_dataset is not None:
            tgt_dataset = PrependTokenDataset(tgt_dataset, tgt_dict.bos())

    eos = None
    if append_source_id:
        src_dataset = AppendTokenDataset(src_dataset, src_dict.index('[{}]'.format(src)))
        if tgt_dataset is not None:
            tgt_dataset = AppendTokenDataset(tgt_dataset, tgt_dict.index('[{}]'.format(tgt)))
        eos = tgt_dict.index('[{}]'.format(tgt))

    align_dataset = None
    if load_alignments:
        align_path = os.path.join(data_path, '{}.align.{}-{}'.format(split, src, tgt))
        if indexed_dataset.dataset_exists(align_path, impl=dataset_impl):
            align_dataset = data_utils.load_indexed_dataset(align_path, None, dataset_impl)

    tgt_dataset_sizes = tgt_dataset.sizes if tgt_dataset is not None else None
    return ContrastiveLanguagePairDataset(
        src_dataset, src_dataset.sizes, src_dict,
        tgt_dataset, tgt_dataset_sizes, tgt_dict,
        left_pad_source=left_pad_source,
        left_pad_target=left_pad_target,
        max_source_positions=max_source_positions,
        max_target_positions=max_target_positions,
        align_dataset=align_dataset, eos=eos
    )


class ContrastiveLanguagePairDataset(LanguagePairDataset):
  def __init__(self, src, src_sizes, src_dict, tgt=None, tgt_sizes=None, tgt_dict=None, label=None, label_dict=None, label_sizes=None, left_pad_source=True,
               left_pad_target=False, max_source_positions=1024, max_target_positions=1024, shuffle=True,
               input_feeding=True, remove_eos_from_source=False, append_eos_to_target=False, align_dataset=None,
               append_bos=False, eos=None):
    super().__init__(src, src_sizes, src_dict, tgt, tgt_sizes, tgt_dict, left_pad_source, left_pad_target,
                     max_source_positions, max_target_positions, shuffle, input_feeding, remove_eos_from_source,
                     append_eos_to_target, align_dataset, append_bos, eos)
    self.label = label
    self.label_dict = label_dict
    self.label_sizes = label_sizes

  def update_label_dataset(self, label, label_dict, label_sizes):
    self.label = label
    self.label_dict = label_dict
    self.label_sizes = label_sizes

  def __getitem__(self, index):
      item = super().__getitem__(index)
      if self.label is not None:
        label_item = self.label[index]
        item.update(label=label_item)
      return item

  def collater(self, samples):
    tran_batch = super().collater(samples)
    labels = torch.LongTensor([s['label'] for s in samples])
    cons_batch = {
      'id': tran_batch['id'],
      'nsentences': tran_batch['nsentences'],
      'ntokens': tran_batch['ntokens'],
      'net_input': {
        'src_tokens': tran_batch['net_input']['src_tokens'],
        'tgt_tokens': tran_batch['target'],
      },
      'label': labels,
    }
    return cons_batch
