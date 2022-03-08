import math

import torch

from fairseq import utils
from fairseq.criterions import FairseqCriterion, register_criterion
from fairseq.logging import metrics

import torch.nn.functional as F

@register_criterion('nt_xent')
class NTXEntCriterion(FairseqCriterion):

    def __init__(self, task, sentence_avg):
        super().__init__(task)
        self.sentence_avg = sentence_avg
        self.temperature = task.args.temperature

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        net_output = model(**sample['net_input'])

        # calculate similarity between src_proj and tgt_proj
        src_rep = net_output['src_output']
        tgt_rep = net_output['tgt_output']
        device = src_rep.device
        bsz, hidden_size = src_rep.size()
        src_rep = src_rep.unsqueeze(1).expand(-1, bsz, -1)
        src_transpose_rep = src_rep.permute(1,0,2)
        tgt_rep = tgt_rep.unsqueeze(1).expand(-1, bsz, -1)
        tgt_transpose_rep = tgt_rep.permute(1,0,2)

        src_tgt_sim = F.cosine_similarity(src_rep, tgt_transpose_rep, dim=2) / self.temperature
        src_src_sim = F.cosine_similarity(src_rep, src_transpose_rep, dim=2) / self.temperature
        tgt_tgt_sim = F.cosine_similarity(tgt_rep, tgt_transpose_rep, dim=2) / self.temperature

        src_logit = torch.cat([src_tgt_sim,src_src_sim], 1)
        tgt_logit = torch.cat([src_tgt_sim,tgt_tgt_sim], 1)
        logit = torch.cat([src_logit,tgt_logit], 0).to(device)

        src_labels = self._make_labels(bsz)
        labels = torch.cat([src_labels, src_labels], 0).to(device)

        src_mask = torch.zeros([bsz, bsz])
        src_mask = src_mask.fill_diagonal_(1)
        mask = torch.cat([torch.zeros([bsz, bsz]), src_mask],1)
        mask = torch.cat([mask, mask], 0).to(device)

        num_sample = sample['nsentences'] * 2
        num_sents = sample['nsentences']
        loss, _ = self.compute_loss(logit, labels, mask, reduce=reduce)
        sample_size = num_sample
        logging_output = {
            'loss': loss.data,
            'nsents': num_sents,
            'nsamples': num_sample,
            'sample_size': sample_size,
        }
        return loss, sample_size, logging_output

    def _make_labels(self, size):
        mask = torch.ones([size])
        return (torch.cumsum(mask, dim=0).type_as(mask)).long() - 1

    def masked_log_softmax(self, vector: torch.Tensor, mask: torch.Tensor, dim: int = -1) -> torch.Tensor:
        """
        ``torch.nn.functional.log_softmax(vector)`` does not work if some elements of ``vector`` should be
        masked.  This performs a log_softmax on just the non-masked portions of ``vector``.  Passing
        ``None`` in for the mask is also acceptable; you'll just get a regular log_softmax.
        ``vector`` can have an arbitrary number of dimensions; the only requirement is that ``mask`` is
        broadcastable to ``vector's`` shape.  If ``mask`` has fewer dimensions than ``vector``, we will
        unsqueeze on dimension 1 until they match.  If you need a different unsqueezing of your mask,
        do it yourself before passing the mask into this function.
        In the case that the input vector is completely masked, the return value of this function is
        arbitrary, but not ``nan``.  You should be masking the result of whatever computation comes out
        of this in that case, anyway, so the specific values returned shouldn't matter.  Also, the way
        that we deal with this case relies on having single-precision floats; mixing half-precision
        floats with fully-masked vectors will likely give you ``nans``.
        If your logits are all extremely negative (i.e., the max value in your logit vector is -50 or
        lower), the way we handle masking here could mess you up.  But if you've got logit values that
        extreme, you've got bigger problems than this.
        """
        if mask is not None:
            mask = mask.float().to(vector.device)
            while mask.dim() < vector.dim():
                mask = mask.unsqueeze(1)
            # vector + mask.log() is an easy way to zero out masked elements in logspace, but it
            # results in nans when the whole vector is masked.  We need a very small value instead of a
            # zero in the mask for these cases.  log(1 + 1e-45) is still basically 0, so we can safely
            # just add 1e-45 before calling mask.log().  We use 1e-45 because 1e-46 is so small it
            # becomes 0 - this is just the smallest value we can actually use.
            vector = vector + (mask + 1e-45).log()
        return torch.nn.functional.log_softmax(vector, dim=dim)

    def compute_loss(self, logit, target, mask, reduce=True):
        lprobs = self.masked_log_softmax(logit, mask)
        lprobs = lprobs.view(-1, lprobs.size(-1))
        loss = F.nll_loss(
            lprobs,
            target,
            reduction='sum' if reduce else 'none',
        )
        return loss, loss

    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get('loss', 0) for log in logging_outputs)
        nsents = sum(log.get('nsents', 0) for log in logging_outputs)
        sample_size = sum(log.get('sample_size', 0) for log in logging_outputs)

        metrics.log_scalar('loss', loss_sum / sample_size / math.log(2), sample_size, round=3)
        if sample_size != nsents:
            metrics.log_scalar('nll_loss', loss_sum / nsents / math.log(2), nsents, round=3)
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
