from fairseq.models import register_model_architecture

from .models import hf_distill_bert, contrastive_hf_distill_bert
from .tasks import translation_joint_adapt, dummy_task, \
   sentence_prediction_from_pretrained_bert, constrastive_nmt_from_pretrained_bert, \
   sentence_prediction_from_contrastive_bert
from .criterions import contrastive, joint_adapt_cross_entropy,sentence_prediction_with_adaptive

@register_model_architecture('hf_distill_bert', 'hf_distill_bert')
def distil_mbert_architecture(args):
   args.pretrained_model = getattr(args, 'pretrained_model', 'distilbert-base-multilingual-cased')
   args.max_positions = getattr(args, 'max_position', 1024)
   hf_distill_bert.base_architecture(args)

@register_model_architecture('contrastive_distill_bert', 'contrastive_distill_bert')
def distil_mbert_architecture(args):
   args.pretrained_model = getattr(args, 'pretrained_model', 'distilbert-base-multilingual-cased')
   args.max_positions = getattr(args, 'max_position', 1024)
   args.adaptive_hidden_size = getattr(args, 'adaptive_hidden_size', 128)
   args.adaptive_layers = getattr(args, 'adaptive_layers', 2)
   args.adaptive_activation_fn = getattr(args, 'adaptive_activation_fn', 'gelu')
   args.adaptive_projection_size = getattr(args, 'adaptive_projection_size', 128)
   args.freeze_encoder = getattr(args, 'freeze_encoder', True)
   contrastive_hf_distill_bert.base_architecture(args)