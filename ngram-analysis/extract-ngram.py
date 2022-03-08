from nltk import ngrams, word_tokenize
import argparse
import logging
from collections import Counter

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

def cli_main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--input', required=True, help="Corpus input")
  parser.add_argument('--output', required=True, help='Output ngram')
  parser.add_argument('--n', default=1, type=int, help="n value in n-gram")
  args = parser.parse_args()

  ngram_counter = Counter()
  with open(args.input, 'r') as fin:
    for line in fin:
      line = line.strip()
      grams = ngrams(word_tokenize(line.lower()), args.n)
      n_grams = (' '.join(item) for item in grams)
      ngram_counter.update(n_grams)

  ngram_counter = sorted(ngram_counter.items())
  with open(args.output, 'w') as fout:
    for key, value in ngram_counter:
      fout.write("{}\t{}\n".format(key, value))

if __name__ == '__main__':
  cli_main()