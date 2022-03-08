import math

import torch

from fairseq import utils
from fairseq.criterions import FairseqCriterion, register_criterion
from fairseq.logging import metrics

import torch.nn.functional as F


def label_smoothed_nll_loss(lprobs, target, epsilon, ignore_index=None, reduce=True):
    if target.dim() == lprobs.dim() - 1:
        target = target.unsqueeze(-1)
    nll_loss = -lprobs.gather(dim=-1, index=target)
    smooth_loss = -lprobs.sum(dim=-1, keepdim=True)
    if ignore_index is not None:
        pad_mask = target.eq(ignore_index)
        nll_loss.masked_fill_(pad_mask, 0.)
        smooth_loss.masked_fill_(pad_mask, 0.)
    else:
        nll_loss = nll_loss.squeeze(-1)
        smooth_loss = smooth_loss.squeeze(-1)
    if reduce:
        nll_loss = nll_loss.sum()
        smooth_loss = smooth_loss.sum()
    eps_i = epsilon / lprobs.size(-1)
    loss = (1. - epsilon) * nll_loss + eps_i * smooth_loss
    return loss, nll_loss


@register_criterion('joint_domain_adapt_xent_with_smoothing')
class JointAdaptXentCriterion(FairseqCriterion):

    def __init__(self, task, sentence_avg, lambda_1, lambda_2, lambda_3, label_smoothing=0.1,
                 disable_src_loss=False, disable_tgt_loss=False):
        super().__init__(task)
        self.sentence_avg = sentence_avg
        self.lambda_1 = lambda_1
        self.lambda_2 = lambda_2
        self.lambda_3 = lambda_3
        self.eps = label_smoothing
        self.disable_src_loss = disable_src_loss
        self.disable_tgt_loss = disable_tgt_loss

    @staticmethod
    def add_args(parser):
        # fmt: off
        parser.add_argument('--lambda_1', default=1, type=float, metavar='D',
                            help='weight for the NMT loss')
        parser.add_argument('--lambda_2', default=1, type=float, metavar='D',
                            help='weight for the discriminative src loss')
        parser.add_argument('--lambda_3', default=1, type=float, metavar='D',
                            help='weight for the discriminative tgt loss')
        parser.add_argument('--label-smoothing', default=0.1, type=float, metavar='D',
                            help='weight for the discriminative loss')
        parser.add_argument('--disable-src-loss', action="store_true", default=False,
                            help='Disable source discriminative loss')
        parser.add_argument('--disable-tgt-loss', action="store_true", default=False,
                            help='Disable target discriminative loss')
        # fmt: on

    def forward(self, model, sample, reduce=True):
        net_output = model(**sample['net_input'])
        bsz = sample['nsentences']
        nmt_mask = None
        num_nmt_sentences = bsz
        num_tokens = sample['ntokens']
        if 'nmt_src_masks' in sample:
            nmt_src_mask = sample['nmt_src_masks']
            nmt_tgt_mask = sample['nmt_tgt_masks']
            num_nmt_sentences = nmt_src_mask.sum() + nmt_tgt_mask.sum()
            num_tokens = sample['tgt_lengths'] * (nmt_src_mask + nmt_tgt_mask)
            num_tokens = num_tokens.sum()
            nmt_mask = nmt_src_mask + nmt_tgt_mask * self.lambda_1
        src_mask = None
        num_disc_sentences = bsz
        if 'src_masks' in sample:
            src_mask = sample['src_masks']
            num_disc_sentences = src_mask.sum()
        nmt_loss, nmt_nll_loss = self.compute_loss_with_label_smoothed(model.nmt,
                                        net_output['nmt_output'], sample['target'].view(-1),
                                        reduce=False, mask=nmt_mask, batch_size=bsz)
        if self.disable_src_loss:
            src_loss = 0
        else:
            src_loss, _ = self.compute_classification_loss(model, net_output['disc_output'], sample['labels'].view(-1),
                                                        mask=src_mask)
            loss += self.lambda_2 * src_loss
            src_loss = src_loss.data
        if self.disable_tgt_loss:
            tgt_loss = 0
        else:
            tgt_loss, _ = self.compute_classification_loss(model, net_output['tgt_disc_output'], sample['labels'].view(-1))
            loss += self.lambda_3 * tgt_loss
            tgt_loss = tgt_loss.data
        sample_size = sample['target'].size(0)
        logging_output = {
            'loss': loss.data,
            'nll_loss': nmt_nll_loss.data,
            'nmt_loss': nmt_loss,
            'disc_loss': self.lambda_2 * src_loss + self.lambda_3 * tgt_loss,
            'src_loss': src_loss,
            'tgt_loss': tgt_loss,
            'ntokens': num_tokens,
            'nsentences': num_nmt_sentences,
            'ndisc_sentences': num_disc_sentences,
            'sample_size': sample_size,
        }
        return loss, sample_size, logging_output

    def compute_loss_with_label_smoothed(self, model, net_output, target, mask=None, reduce=True, batch_size=None):
        lprobs = model.get_normalized_probs(net_output, log_probs=True)
        lprobs = lprobs.view(-1, lprobs.size(-1))
        reduce = reduce and (mask is not None)
        loss, nll_loss = label_smoothed_nll_loss(
            lprobs, target, self.eps, ignore_index=self.padding_idx, reduce=reduce,
        )
        if mask is not None:
            loss = loss.view(batch_size, -1).sum(1) * mask
            nll_loss = nll_loss.view(batch_size, -1).sum(1) * mask
            return loss.sum(), nll_loss.sum()
        return loss, nll_loss

    def compute_classification_loss(self, model, logits, targets, mask=None):
        lprobs = F.log_softmax(logits, dim=-1, dtype=torch.float32)
        lprobs = lprobs.view(-1, lprobs.size(-1))
        targets = targets.view(-1)
        # if targets.dim() == lprobs.dim() - 1:
        #     lprobs = lprobs.squeeze(0)
        if mask is None:
            loss = F.nll_loss(lprobs, targets, reduction='sum')
        else:
            mask = mask.view(-1)
            loss = F.nll_loss(lprobs, targets, reduce=False) * mask
            loss = loss.sum()
        return loss, loss

    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get('loss', 0) for log in logging_outputs)
        ntokens = sum(log.get('ntokens', 0) for log in logging_outputs)
        sample_size = sum(log.get('sample_size', 0) for log in logging_outputs)
        nll_loss_sum = sum(log.get('nll_loss', 0) for log in logging_outputs)
        nmt_sample_size = sum(log.get('nsentences', 0) for log in logging_outputs)
        disc_sample_size = sum(log.get('ndisc_sentences', nmt_sample_size) for log in logging_outputs)
        nmt_loss_sum = sum(log.get('nmt_loss', 0) for log in logging_outputs)
        disc_loss_sum = sum(log.get('disc_loss', 0) for log in logging_outputs)

        metrics.log_scalar('loss', loss_sum / sample_size / math.log(2), sample_size, round=3)
        metrics.log_scalar('nmt_loss', nmt_loss_sum / nmt_sample_size / math.log(2), sample_size, round=3)
        metrics.log_scalar('disc_loss', disc_loss_sum / disc_sample_size / math.log(2), sample_size, round=3)
        if sample_size != ntokens:
            metrics.log_scalar('nll_loss', loss_sum / ntokens / math.log(2), ntokens, round=3)
            metrics.log_scalar('nll_nmt_loss', nll_loss_sum / ntokens / math.log(2), ntokens, round=3)
            metrics.log_scalar('nll_disc_loss', disc_loss_sum / disc_sample_size / math.log(2), ntokens, round=3)
            metrics.log_derived('ppl', lambda meters: utils.get_perplexity(meters['nll_loss'].avg))
        else:
            metrics.log_derived('ppl', lambda meters: utils.get_perplexity(meters['loss'].avg))

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True
