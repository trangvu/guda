import logging
import os

from fairseq.data.dictionary import PrebuiltDictionary
from fairseq.tasks import register_task
from fairseq.tasks.sentence_prediction import SentencePredictionTask

logger = logging.getLogger(__name__)


@register_task("domain_classification")
class DomainClassificationTask(SentencePredictionTask):
    @staticmethod
    def add_args(parser):
        SentencePredictionTask.add_args(parser)

        """Add task-specific arguments to the parser."""
        parser.add_argument('data', metavar='FILE',
                            help='file prefix for data')
        parser.add_argument('--num-classes', type=int, default=-1,
                            help='number of classes or regression targets')
        parser.add_argument('--init-token', type=int, default=None,
                            help='add token at the beginning of each batch item')
        parser.add_argument('--separator-token', type=int, default=None,
                            help='add separator token between inputs')
        parser.add_argument('--regression-target', action='store_true', default=False)
        parser.add_argument('--no-shuffle', action='store_true', default=False)
        parser.add_argument('--shorten-method', default='none',
                            choices=['none', 'truncate', 'random_crop'],
                            help='if not none, shorten sequences that exceed --tokens-per-sample')
        parser.add_argument('--shorten-data-split-whitelist', default='',
                            help='comma-separated list of dataset splits to apply shortening to, '
                                 'e.g., "train,valid" (default: all dataset splits)')
        parser.add_argument('--add-prev-output-tokens', action='store_true', default=False,
                            help='add prev_output_tokens to sample, used for encoder-decoder arch')
        parser.add_argument('--config-file', help="Config file")
        parser.add_argument('--offset-positions-by-padding', default=True, action='store_false',
                            help='Offset positions by padding')

    def __init__(self, args, data_dictionary, label_dictionary):
        super().__init__(args, data_dictionary, label_dictionary)

    @classmethod
    def setup_task(cls, args, **kwargs):
        assert args.num_classes > 0, 'Must set --num-classes'

        # load data dictionary
        data_dict = PrebuiltDictionary.load(os.path.join(args.data, 'input0', 'dict.txt'))
        data_dict.init_special_tokens()
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
        return cls(args, data_dict, label_dict)


