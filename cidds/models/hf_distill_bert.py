'''
Created by trangvu on 21/11/21
'''
import logging

from fairseq import utils
from fairseq.models import register_model, FairseqEncoderModel, FairseqEncoder
import torch.nn as nn
import torch.nn.functional as F
from fairseq.models.masked_lm import MaskedLMModel
from transformers import DistilBertModel

from ..data.pretrained_dictionary import PretrainedDictionary

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

@register_model("hf_distill_bert")
class HuggingFacePretrainedDistillBertModel(FairseqEncoderModel):
  def __init__(self, args, encoder):
    super().__init__(encoder)
    self.args = args
    self.classification_heads = nn.ModuleDict()

  @staticmethod
  def add_args(parser):
    """Add model-specific arguments to the parser."""
    MaskedLMModel.add_args(parser)
    # fmt: off
    parser.add_argument(
      "--pretrained-model",
      type=str,
      metavar="STR",
      help="HF pretrained model name",
    )

    parser.add_argument('--pooler-dropout', type=float, metavar='D', default=0.1,
                        help='dropout probability in the masked_lm pooler layers')
    parser.add_argument('--load-checkpoint-heads', action='store_true',
                        help='(re-)register and load heads when loading checkpoints')
    # fmt: on

  def forward(
        self,
        src_tokens,
        src_lengths,
        return_all_hiddens: bool = True,
        features_only: bool = False,
        classification_head_name=None, **kwargs
    ):
      if classification_head_name is not None:
        features_only = True
      x, extra = self.encoder(src_tokens, features_only, return_all_hiddens, **kwargs)

      if classification_head_name is not None:
        x = self.classification_heads[classification_head_name](x)
      return x, extra

  def get_normalized_probs(self, net_output, log_probs, sample=None):
    """Get normalized probabilities (or log probs) from a net's output."""
    logits = net_output[0].float()
    if log_probs:
      return F.log_softmax(logits, dim=-1)
    else:
      return F.softmax(logits, dim=-1)

  def register_classification_head(self, name, num_classes=None, inner_dim=None, **kwargs):
    """Register a classification head."""
    if name in self.classification_heads:
      prev_num_classes = self.classification_heads[name].out_proj.out_features
      prev_inner_dim = self.classification_heads[name].dense.out_features
      if num_classes != prev_num_classes or inner_dim != prev_inner_dim:
        logger.warning(
          're-registering head "{}" with num_classes {} (prev: {}) '
          'and inner_dim {} (prev: {})'.format(
            name, num_classes, prev_num_classes, inner_dim, prev_inner_dim
          )
        )

    self.classification_heads[name] = BertClassificationHead(
      self.args.encoder_embed_dim,
      inner_dim or self.args.encoder_embed_dim,
      num_classes,
      self.args.pooler_activation_fn,
      self.args.pooler_dropout
    )

  @classmethod
  def build_model(cls, args, task, cls_dictionary=PretrainedDictionary):
    """Build a new model instance."""

    # make sure all arguments are present
    base_architecture(args)

    if not hasattr(args, 'max_positions'):
      args.max_positions = args.tokens_per_sample
    logger.info(args)

    encoder = HuggingFacePretrainedDistillBertEncoder(args, task.source_dictionary)
    return cls(args, encoder)

class HuggingFacePretrainedDistillBertEncoder(FairseqEncoder):
  def __init__(self, args, dictionary):
    try:
      from transformers import DistilBertConfig, DistilBertForSequenceClassification
    except ImportError:
      raise ImportError(
        "\n\nPlease install huggingface/transformers with:"
        "\n\n  pip install transformers"
      )
    super().__init__(dictionary)
    self.args = args
    self.encoder = DistilBertModel.from_pretrained(args.pretrained_model)
    self.padding_idx = dictionary.pad()


  def forward(self, src_tokens, features_only=False, return_all_hiddens=False, src_lengths=None, **kwargs):
    encoder_padding_mask = (~src_tokens.eq(self.padding_idx)).float()
    encoder_outputs = self.encoder(input_ids=src_tokens,
                                   attention_mask=encoder_padding_mask)
    hidden_state = encoder_outputs[0]
    pooled_output = hidden_state[:,0]
    return pooled_output, {
      "inner_states": hidden_state if return_all_hiddens else None
    }


class BertClassificationHead(nn.Module):
  """Head for sentence-level classification tasks."""

  def __init__(
    self,
    input_dim,
    inner_dim,
    num_classes,
    activation_fn,
    pooler_dropout,
  ):
    super().__init__()
    self.dense = nn.Linear(input_dim, inner_dim)
    self.activation_fn = utils.get_activation_fn(activation_fn)
    self.dropout = nn.Dropout(p=pooler_dropout)
    self.out_proj = nn.Linear(inner_dim, num_classes)

  def forward(self, features, **kwargs):
    x = features
    x = self.dropout(x)
    x = self.dense(x)
    x = self.activation_fn(x)
    x = self.dropout(x)
    x = self.out_proj(x)
    return x

def base_architecture(args):
  args.pooler_activation_fn = getattr(args, 'pooler_activation_fn', 'tanh')
  args.max_positions = getattr(args, 'max_positions', 512)
  args.encoder_embed_dim = getattr(args, 'encoder_embed_dim', 768)
  args.share_encoder_input_output_embed = getattr(
    args, 'share_encoder_input_output_embed', True)
  args.no_token_positional_embeddings = getattr(
    args, 'no_token_positional_embeddings', False)
  args.encoder_learned_pos = getattr(args, 'encoder_learned_pos', True)
  args.num_segment = getattr(args, 'num_segment', 2)

  args.encoder_layers = getattr(args, 'encoder_layers', 12)

  args.encoder_attention_heads = getattr(args, 'encoder_attention_heads', 12)
  args.encoder_ffn_embed_dim = getattr(args, 'encoder_ffn_embed_dim', 3072)

  args.sentence_class_num = getattr(args, 'sentence_class_num', 2)
  args.sent_loss = getattr(args, 'sent_loss', True)

  args.apply_bert_init = getattr(args, 'apply_bert_init', False)

  args.activation_fn = getattr(args, 'activation_fn', 'gelu')
  args.pooler_activation_fn = getattr(args, 'pooler_activation_fn', 'tanh')
  args.encoder_normalize_before = getattr(args, 'encoder_normalize_before', True)
