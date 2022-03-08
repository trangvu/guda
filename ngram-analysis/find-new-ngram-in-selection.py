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
  parser.add_argument('--input', required=True, help="Selection corpus")
  parser.add_argument('--new-ngram', required=True, help="List of new ngram")
  parser.add_argument('--output', required=True, help='Output ngram')
  parser.add_argument('--n', default=1, type=int, help="n value in n-gram")
  args = parser.parse_args()

  select_ngrams = get_ngrams(args.input, args.n)

  new_ngram = []
  with open(args.new_ngram, 'r') as fin:
    for line in fin:
      new_ngram.append(line.strip())

  new_ngram = set(new_ngram)
  inter_grams = new_ngram.intersection(select_ngrams)
  logger.info("Number of {}-grams: ".format(args.n))
  logger.info(" * In selection data: {}".format(len(select_ngrams)))
  logger.info(" * New grams (appear in new domain but not in old domain): {}".format(len(new_ngram)))
  logger.info(" * Intersection: {}".format(len(inter_grams)))
  with open(args.output, 'w') as fout:
    for item in inter_grams:
      fout.write("{}\n".format(item))

if __name__ == '__main__':
  cli_main()