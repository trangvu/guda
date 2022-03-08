import json
import logging
import os
from argparse import Namespace

import numpy as np
import torch
from torch.serialization import default_restore_location

from fairseq import utils
from fairseq.criterions.cross_entropy import CrossEntropyCriterion
from fairseq.data import encoders, data_utils, AppendTokenDataset, TruncateDataset, StripTokenDataset, RawLabelDataset, \
    ConcatDataset, NestedDictionaryDataset, IdDataset, NumelDataset, NumSamplesDataset, PadDataset, SortDataset, \
    Dictionary, FairseqDataset, OffsetTokensDataset
from fairseq.file_io import PathManager

from ..models.encoder_decoder_with_domain_discriminator import EncoderDecoderWithDomainDisciriminator
from fairseq.tasks import register_task
from fairseq.tasks.translation import TranslationTask

logger = logging.getLogger(__name__)
EVAL_BLEU_ORDER = 4

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

@register_task("da_translation")
class JointAdaptTransaltion(TranslationTask):
    @staticmethod
    def add_args(parser):
        TranslationTask.add_args(parser)
        parser.add_argument('--discriminator-hidden-size', type=int, metavar='N',
                            help='hidden size of discriminator layers')
        parser.add_argument('--discriminator-layers', type=int, metavar='N',
                            help='num discriminator layers')
        parser.add_argument('--discriminator-activation-fn',
                            choices=utils.get_available_activation_fns(),
                            help='activation function to use')
        parser.add_argument('--discriminator-dropout', type=float, metavar='D', default=0.1,
                            help='dropout probability in the masked_lm discriminator layers')
        # parser.add_argument('--unlabel-subset', type=str, default='mono',
        #                     help='Unlabel subset')
        parser.add_argument('--max-positions', type=int, default=512,
                            help='Max position')
        parser.add_argument('--pretrained-nmt', type=str, help='Path to pretrained NMT model')
        parser.add_argument('--label-subset',  type=str, default="label")

    def __init__(self, args, src_dict, tgt_dict):
        super().__init__(args, src_dict, tgt_dict)
        self.indomain_offset = -1
        self.source_offset = -1

        # Find accumulate dataset



    def build_model(self, args):
        # build nmt model
        nmt_model = super().build_model(args)
        if getattr(args, 'eval_bleu', False):
            assert getattr(args, 'eval_bleu_detok', None) is not None, (
                '--eval-bleu-detok is required if using --eval-bleu; '
                'try --eval-bleu-detok=moses (or --eval-bleu-detok=space '
                'to disable detokenization, e.g., when using sentencepiece)'
            )
            detok_args = json.loads(getattr(args, 'eval_bleu_detok_args', '{}') or '{}')
            self.tokenizer = encoders.build_tokenizer(Namespace(
                tokenizer=getattr(args, 'eval_bleu_detok', None),
                **detok_args
            ))

            gen_args = json.loads(getattr(args, 'eval_bleu_args', '{}') or '{}')
            self.sequence_generator = self.build_generator([nmt_model], Namespace(**gen_args))
        model = EncoderDecoderWithDomainDisciriminator(args, nmt_model)
        if hasattr(args, 'pretrained_nmt') and args.pretrained_nmt:
            model_checkpoint = args.pretrained_nmt
            logger.info("Load NMT models from {}".format(model_checkpoint))
            model.nmt = load_pretrained_model(model.nmt, model_checkpoint)
        return model


    def load_dataset(self, split, epoch=1, combine=False, **kwargs):
        super().load_dataset(split, epoch, combine, **kwargs)

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
        if split == "train":
            # Load label subset
            label_dataset = None
            if not(self.args.disable_tgt_loss and self.args.disable_src_loss):
                label_path = os.path.join(self.args.data, self.args.label_subset)
                dict_label_path = os.path.join(self.args.data, self.args.label_subset, 'dict.txt')
                label_dict = Dictionary.load(dict_label_path)
                label_dict.add_symbol('<mask>')
                label_dataset = make_dataset(label_path, label_dict)
                label_dataset = OffsetTokensDataset(
                            StripTokenDataset(
                                label_dataset,
                                id_to_strip=label_dict.eos(),
                            ),
                            offset=-label_dict.nspecial,
                        )
            lang_pair_dataset = self.datasets[split]
            cum_size = None
            if isinstance(lang_pair_dataset, ConcatDataset):
                cum_size = lang_pair_dataset.src.cumulative_sizes
            dataset = LanguagePairWithDomainLabelDataset(lang_pair_dataset, label_dataset, cum_size)
            self.datasets[split] = dataset
            # # Load old and new domain data for source and target task
            # old_domain_src = self.datasets[split].src
            # old_domain_tgt = self.datasets[split].tgt
            # src, tgt = self.args.source_lang, self.args.target_lang
            # src_dict = self.src_dict
            # tgt_dict = self.tgt_dict
            # unlabel_split = self.args.unlabel_subset
            # prefix = os.path.join(self.args.data, '{}.{}-{}.'.format(unlabel_split, src, tgt))
            # src_dataset = data_utils.load_indexed_dataset(prefix + src, src_dict, self.args.dataset_impl)
            # if self.args.truncate_source:
            #     src_dataset = AppendTokenDataset(
            #         TruncateDataset(
            #             StripTokenDataset(src_dataset, src_dict.eos()),
            #             self.args.max_source_positions - 1,
            #         ),
            #         src_dict.eos(),
            #     )
            # tgt_dataset = data_utils.load_indexed_dataset(prefix + tgt, tgt_dict, self.args.dataset_impl)
            #
            # old_domain_size = len(old_domain_src)
            # old_domain_labels = [0] * old_domain_size
            # old_domain_label_dataset = RawLabelDataset(old_domain_labels)
            # new_domain_source_label_dataset = RawLabelDataset([1] * len(src_dataset))
            # new_domain_target_label_dataset = RawLabelDataset([1] * len(tgt_dataset))
            # sample_ratios = [1,1]
            # unabel_source_dataset = ConcatDataset([old_domain_src, src_dataset], sample_ratios)
            # unabel_target_dataset = ConcatDataset([old_domain_tgt, tgt_dataset], sample_ratios)
            # source_labels = ConcatDataset([old_domain_label_dataset, new_domain_source_label_dataset], sample_ratios)
            # target_labels = ConcatDataset([old_domain_label_dataset, new_domain_target_label_dataset], sample_ratios)
            #
            # print(" * NMT dataset size {} - mono source size {} - mono target size {}".
            #              format(len(old_domain_label_dataset), len(source_labels), len(target_labels)))
            #
            # with data_utils.numpy_seed(self.args.seed + epoch):
            #     src_shuffle = np.random.permutation(len(unabel_source_dataset))
            #     tgt_shuffle = np.random.permutation(len(unabel_target_dataset))
            #
            # self.datasets["unlabel_source"] = SortDataset(NestedDictionaryDataset(
            #     {
            #         'id': IdDataset(),
            #         'net_input': {
            #             'src_tokens': PadDataset(unabel_source_dataset,
            #                             pad_idx=src_dict.pad(),
            #                             left_pad=False),
            #             'src_lengths': NumelDataset(unabel_source_dataset, reduce=False)
            #         },
            #         'nsentences': NumSamplesDataset(),
            #         'ntokens': NumelDataset(unabel_source_dataset, reduce=True),
            #         'target': source_labels
            #     },
            #     sizes=[unabel_source_dataset.sizes],
            #     ),
            #     sort_order=[
            #         src_shuffle,
            #         unabel_source_dataset.sizes,],
            # )
            #
            # self.datasets["unlabel_target"] = SortDataset(NestedDictionaryDataset(
            #     {
            #         'id': IdDataset(),
            #         'net_input': {
            #             'src_tokens': PadDataset(unabel_target_dataset,
            #                             pad_idx=src_dict.pad(),
            #                             left_pad=False),
            #             'src_lengths': NumelDataset(unabel_target_dataset, reduce=False)
            #         },
            #         'nsentences': NumSamplesDataset(),
            #         'ntokens': NumelDataset(unabel_target_dataset, reduce=True),
            #         'target': target_labels
            #     },
            #     sizes=[unabel_target_dataset.sizes],),
            #     sort_order=[
            #         tgt_shuffle,
            #         unabel_target_dataset.sizes,],
            # )

    def valid_step(self, sample, model, criterion):
        model = model.nmt
        nmt_criterion = CrossEntropyCriterion(self, self.args.sentence_avg)
        loss, sample_size, logging_output = super().valid_step(sample, model, nmt_criterion)
        if self.args.eval_bleu:
            bleu = self._inference_with_bleu(self.sequence_generator, sample, model)
            logging_output['_bleu_sys_len'] = bleu.sys_len
            logging_output['_bleu_ref_len'] = bleu.ref_len
            # we split counts into separate entries so that they can be
            # summed efficiently across workers using fast-stat-sync
            assert len(bleu.counts) == EVAL_BLEU_ORDER
            for i in range(EVAL_BLEU_ORDER):
                logging_output['_bleu_counts_' + str(i)] = bleu.counts[i]
                logging_output['_bleu_totals_' + str(i)] = bleu.totals[i]
        return loss, sample_size, logging_output

    def build_generator(self, models, args):
        nmt_models = [m.nmt for m in models]
        return super().build_generator(nmt_models, args)

class LanguagePairWithDomainLabelDataset(FairseqDataset):
    def __init__(self, lang_pair, label, offset=None):
        # offset is a list of size 3, storing the offset of NMT old domain, backtranslate in-domain and forward translate  indomain
        self.offset = offset
        self.lang_pair = lang_pair
        self.label = label

    def __len__(self):
        return len(self.lang_pair)

    def __getitem__(self, index):
        item = self.lang_pair[index]
        if self.label is not None:
            item.update(label=self.label[index])
        else:
            item.update(label=-1)
        nmt_src_mask = 1
        nmt_tgt_mask = 0
        src_mask = 1
        if self.offset:
            if index > self.offset[1]:
                #forward in-domain, i.e. clean src language, pseudo tgt
                nmt_src_mask = 0
                nmt_tgt_mask = 0
                src_mask = 1
            elif index > self.offset[0]:
                # backtranslate in-domain, i.e. pseudo src language, clean tgt
                nmt_tgt_mask = 1
                nmt_src_mask = 0
                src_mask = 0
        item.update(nmt_src_mask=nmt_src_mask)
        item.update(nmt_tgt_mask=nmt_tgt_mask)
        item.update(src_mask=src_mask)
        return item

    def collater(self, samples):
        batch = self.lang_pair.collater(samples)
        labels = torch.LongTensor([s['label'] for s in samples])
        nmt_src_masks = torch.LongTensor([s['nmt_src_mask'] for s in samples])
        nmt_tgt_masks = torch.LongTensor([s['nmt_tgt_mask'] for s in samples])
        src_masks = torch.LongTensor([s['src_mask'] for s in samples])
        tgt_lengths = torch.LongTensor([s['target'].numel() for s in samples])
        batch.update(labels=labels)
        batch.update(nmt_src_masks=nmt_src_masks)
        batch.update(nmt_tgt_masks=nmt_tgt_masks)
        batch.update(src_masks=src_masks)
        batch.update(tgt_lengths=tgt_lengths)
        return batch

    def num_tokens(self, index):
        return self.lang_pair.num_tokens(index)

    def size(self, index):
        return self.lang_pair.size(index)

    def ordered_indices(self):
        return self.lang_pair.ordered_indices()

    @property
    def supports_prefetch(self):
        return self.lang_pair.supports_prefetch

    def prefetch(self, indices):
        self.lang_pair.prefetch(indices)
        self.label.prefetch(indices)
