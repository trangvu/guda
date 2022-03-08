# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.serialization import default_restore_location

from fairseq import utils
from fairseq.file_io import PathManager
from fairseq.models import (
    FairseqEncoderModel,
    FairseqEncoder,
    register_model,
    register_model_architecture, BaseFairseqModel,
)
from fairseq.models.masked_lm import MaskedLMModel, MaskedLMEncoder

from .hf_distill_bert import HuggingFacePretrainedDistillBertEncoder, BertClassificationHead

logger = logging.getLogger(__name__)

@register_model('contrastive_distill_bert')
class ContrastiveFromPretrainedBert(BaseFairseqModel):
    """
    Class for contrastive learning from pretrained MLM
    """

    def __init__(self, args, encoder):
        super().__init__()
        self.encoder = encoder
        self.args = args
        self.adaptive_layers = nn.ModuleDict()
        self.classification_heads = nn.ModuleDict()

    @staticmethod
    def add_args(parser):
        # Add MaskedLM model args
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
        # Arguments related to adaptive layer g
        parser.add_argument('--adaptive-hidden-size', type=int, metavar='N',
                            help='hidden size of adaptive layers')
        parser.add_argument('--adaptive-layers', type=int, metavar='N',
                            help='num adaptive layers')
        parser.add_argument('--adaptive-layer-name', metavar='N', default='adaptive',
                            help='adaptive layer name')
        parser.add_argument('--adaptive-activation-fn',
                            choices=utils.get_available_activation_fns(),
                            help='activation function to use')
        parser.add_argument('--adaptive-projection-size', type=int, metavar='N',
                            help='output size of adaptive layer')
        parser.add_argument('--load-checkpoint-heads', action='store_true',
                            help='(re-)register and load heads when loading checkpoints')
        parser.add_argument('--freeze-encoder', action='store_true', help='do not train the encoder')
        parser.add_argument('--pretrained-path', help='Load pretrained weights')

        # misc params

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""
        # make sure all arguments are present in older models
        base_architecture(args)

        if not hasattr(args, 'max_positions'):
            args.max_positions = args.tokens_per_sample

        logger.info(args)

        encoder = HuggingFacePretrainedDistillBertEncoder(args, task.source_dictionary)
        if args.freeze_encoder:
          logger.info("Freeze the encoder")
          for p in encoder.parameters():
            p.requires_grad = False
        return cls(args, encoder)

    def register_adaptive_layer(self, name, inner_dim=None, **kwargs):
        """Register an adaptive layer"""
        if name in self.adaptive_layers:
                logger.warning(
                    're-registering adaptive layer "{}"'.format(name)
                )

        self.adaptive_layers[name] = AdaptiveLayer(num_hidden_layers = self.args.adaptive_layers,
                                     input_size = self.args.encoder_embed_dim,
                                     hidden_size = self.args.adaptive_hidden_size,
                                     projection_size = self.args.adaptive_projection_size,
                                     activation_fn= self.args.adaptive_activation_fn)

    def register_classification_head(self, name, num_classes=None, inner_dim=None,
                                     adaptive_layer=True, **kwargs):
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

        input_dim = self.args.encoder_embed_dim
        if adaptive_layer:
            input_dim = self.args.adaptive_projection_size
        self.classification_heads[name] = BertClassificationHead(
            input_dim,
            inner_dim or input_dim,
            num_classes,
            self.args.pooler_activation_fn,
            self.args.pooler_dropout
        )

    def forward(
      self,
      src_tokens,
      tgt_tokens=None,
      return_all_hiddens: bool = True,
      features_only: bool = False,
      source_only: bool = False,
      adaptive_head_name=None,
      classification_head_name=None, **kwargs
    ):
        if adaptive_head_name is not None:
            features_only = True
        x_src, extra = self.encoder(src_tokens, features_only, return_all_hiddens, **kwargs)
        if adaptive_head_name is not None:
          x_src = self.adaptive_layers[adaptive_head_name](x_src)
        if classification_head_name is not None:
            x_src = self.classification_heads[classification_head_name](x_src)
        if not source_only:
            x_tgt, extra = self.encoder(tgt_tokens, features_only, return_all_hiddens, **kwargs)
            if adaptive_head_name is not None:
              x_tgt = self.adaptive_layers[adaptive_head_name](x_tgt)
            if classification_head_name is not None:
                x_tgt = self.classification_heads[classification_head_name](x_tgt)
            net_output = {
                'src_output': x_src,
                'tgt_output': x_tgt
            }
        else:
            net_output = {
                'src_output': x_src
            }
        return net_output


    def max_positions(self):
        return self.encoder.max_positions()

    # def upgrade_state_dict(self, state_dict):
    #     self.encoder.upgrade_state_dict(state_dict)
    #     self.adaptive_layers.upgrade_state_dict(state_dict)
    #
    #     return state_dict

    def get_normalized_probs(self, net_output, log_probs, sample=None):
        """Get normalized probabilities (or log probs) from a net's output."""
        logits = net_output[0].float()
        if log_probs:
            return F.log_softmax(logits, dim=-1)
        else:
            return F.softmax(logits, dim=-1)

class AdaptiveLayer(nn.Module):
    def __init__(self, num_hidden_layers, input_size, hidden_size, projection_size, activation_fn='relu', dropout=0.1):
        super().__init__()
        input_layer = nn.Linear(input_size, hidden_size)
        self.hidden_layers = nn.ModuleList([input_layer])
        activation_fn = utils.get_activation_fn(activation_fn)
        if num_hidden_layers - 1 > 0:
            layer = FeedForwadLayer(hidden_size, hidden_size, activation_fn, dropout)
            self.hidden_layers.extend([deepcopy(layer) for _ in range(num_hidden_layers - 1)])
        self.output_layer =  FeedForwadLayer(hidden_size, projection_size, activation_fn, dropout)

    def forward(self, states):
        hidden_states = states
        for layer_module in self.hidden_layers:
            hidden_states = layer_module(hidden_states)
        projection_output = self.output_layer(hidden_states)
        return projection_output


class FeedForwadLayer(nn.Module):
    def __init__(self, input_size, output_size, activation_fn,  dropout_prob=None):
        super(FeedForwadLayer, self).__init__()
        self.layer = nn.Linear(input_size, output_size)
        self.activate_fn = activation_fn
        if dropout_prob is not None:
            self.dropout = nn.Dropout(p=dropout_prob)
        else:
            self.dropout = None

    def forward(self, states):
        hidden_states = self.layer(states)
        if self.dropout:
            hidden_states = self.dropout(hidden_states)
        hidden_states = self.activate_fn(hidden_states)
        return hidden_states

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
