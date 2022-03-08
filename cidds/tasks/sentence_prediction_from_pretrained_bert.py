'''
Created by trangvu on 12/12/21
'''
import logging

from fairseq.tasks import register_task
from fairseq.tasks.sentence_prediction import SentencePredictionTask

from ..data.pretrained_dictionary import PretrainedDictionary, Dictionary

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

@register_task("sentence_prediction_from_pretrained_bert")
class SentencePredictionFromPretrainedBertTask(SentencePredictionTask):
  @classmethod
  def load_dictionary(cls, args, filename, source=True):
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
