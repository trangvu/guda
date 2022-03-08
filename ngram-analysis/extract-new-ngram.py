from nltk import ngrams, word_tokenize
import argparse
import logging
from collections import Counter

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

def get_ngrams(corpus, n):
  ngram_counter = Counter()
  with open(corpus, 'r') as fin:
    for line in fin:
      line = line.strip()
      grams = ngrams(word_tokenize(line.lower()), n)
      n_grams = (' '.join(item) for item in grams)
      ngram_counter.update(n_grams)
  return set(ngram_counter.keys())

def cli_main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--input-new', required=True, help="Test set of new domain")
  parser.add_argument('--input-old', required=True, help="Test set of old domain")
  parser.add_argument('--output', required=True, help='Output ngram')
  parser.add_argument('--n', default=1, type=int, help="n value in n-gram")
  args = parser.parse_args()

  newdomain_ngrams = get_ngrams(args.input_new, args.n)
  olddomain_ngrams = get_ngrams(args.input_old, args.n)

  inter_grams = newdomain_ngrams.intersection(olddomain_ngrams)
  new_ngrams = newdomain_ngrams - olddomain_ngrams

  logger.info("Number of {}-grams: ".format(args.n))
  logger.info(" * Old domain: {}".format(len(olddomain_ngrams)))
  logger.info(" * New domain: {}".format(len(newdomain_ngrams)))
  logger.info(" * New grams (appear in new domain but not in old domain): {}".format(len(new_ngrams)))
  logger.info(" * Intersection: {}".format(len(inter_grams)))
  with open(args.output, 'w') as fout:
    for item in new_ngrams:
      fout.write("{}\n".format(item))

if __name__ == '__main__':
  cli_main()