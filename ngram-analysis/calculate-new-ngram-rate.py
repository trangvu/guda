from nltk import ngrams, word_tokenize
import argparse
import logging
from collections import Counter

from itertools import zip_longest, count

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

def get_ngrams(ref, src, tgt, n):

  ref_ngram = (' '.join(item) for item in ngrams(word_tokenize(ref.lower()), n))
  src_ngram = (' '.join(item) for item in ngrams(word_tokenize(src.lower()), n))
  tgt_ngram = (' '.join(item) for item in ngrams(word_tokenize(tgt.lower()), n))

  ref_set = set(ref_ngram)
  src_set = set(src_ngram)
  tgt_set = set(tgt_ngram)

  new_gram = ref_set - src_set
  correct_gram = tgt_set.intersection(new_gram)
  return len(new_gram), len(correct_gram)

def cli_main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--reference', required=True, help="Reference translation")
  parser.add_argument('--source_hypo', required=True, help="Hypothesis of source model")
  parser.add_argument('--target_hypo', required=True, help='Hypothesis of target model')
  parser.add_argument('--Output', required=True, help='Output')
  parser.add_argument('--n', default=1, type=int, help="n value in n-gram")
  args = parser.parse_args()

  cnt = 0
  total_new_ngram = 0
  total_correct_ngram = 0
  all_counts = []
  with open(args.reference) as fref, open(args.source_hypo) as fsrc, open(args.target_hypo) as ftgt:
    for index, ref, src, tgt in zip_longest(count(), fref,fsrc, ftgt):
      cnt +=1
      new_ngram, correct_ngram = get_ngrams(ref, src, tgt, args.n)
      all_counts.append((new_ngram, correct_ngram))
      total_new_ngram += new_ngram
      total_correct_ngram += total_correct_ngram



  logger.info("Number of {}-grams: ".format(args.n))
  logger.info(" * New ngram: {}".format(len(total_new_ngram)))
  logger.info(" * Correct ngram: {}".format(len(total_correct_ngram)))
  logger.info(" * Percentage: {}".format(total_correct_ngram/total_new_ngram * 100))
  with open(args.output, 'w') as fout:
    for item in all_counts:
      fout.write("{} {}\n".format(item[0], item[1]))

if __name__ == '__main__':
  cli_main()