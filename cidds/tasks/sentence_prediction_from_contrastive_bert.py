'''
Created by trangvu on 12/12/21
'''
import logging
import os
import torch

from fairseq.file_io import PathManager
from fairseq.tasks import register_task
from torch.serialization import default_restore_location

from .sentence_prediction_from_pretrained_bert import SentencePredictionFromPretrainedBertTask

from ..data.pretrained_dictionary import PretrainedDictionary, Dictionary

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


def load_pretrained_model(model, pretrained_path):
    bexists = PathManager.isfile(pretrained_path)
    if bexists:
        with PathManager.open(pretrained_path, "rb") as f:
            state = torch.load(
                f, map_location=lambda s, l: default_restore_location(s, "cpu")
            )
        logger.info("Load model from {}".format(pretrained_path))
        model.load_state_dict(
            state["model"], strict=True
        )
    else:
        logger.warning("Cannot file checkpoint {}".format(pretrained_path))
        return model
    return model



@register_task("sentence_prediction_from_contrastive_bert")
class SentencePredictionFromContrastiveBertTask(SentencePredictionFromPretrainedBertTask):
  def build_model(self, args):
    from fairseq import models
    model = models.build_model(args, self)
    model.register_adaptive_layer('adaptive')
    if args.pretrained_path:
      logger.info("Load pretrained weight from {}".format(args.pretrained_path))
      load_pretrained_model(model, args.pretrained_path)
    model.register_classification_head(
      getattr(args, 'classification_head_name', 'sentence_classification_head'),
      num_classes=self.args.num_classes,
      adaptive_layer=True
    )
    return model

  @classmethod
  def setup_task(cls, args, **kwargs):
    assert args.num_classes > 0, 'Must set --num-classes'

    # load data dictionary
    data_dict = cls.load_dictionary(
      args,
      os.path.join(args.data, 'input0', 'dict.txt'),
      source=True,
    )
    logger.info('[input] dictionary: {} types'.format(len(data_dict)))

    label_dict = None
    if not args.regression_target:
      # load label dictionary
      label_dict = cls.load_dictionary(
        args,
        os.path.join(args.data, 'label', 'dict.txt'),
        source=False,
      )
      logger.info('[label] dictionary: {} types'.format(len(label_dict)))
    else:
      label_dict = data_dict
    return SentencePredictionFromContrastiveBertTask(args, data_dict, label_dict)