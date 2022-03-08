# split the data to best 50k, 500k, 1m, 1.5m
import argparse
import logging
import random
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import operator
import os

from tqdm import tqdm

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

def write_topk_to_disc(scored_data, path_prefix, src, trg, k):
    logger.info("Write top {} to disc".format(k))
    with open(path_prefix + '.top{}'.format(k) + '.{}'.format(src), 'w') as src_file, \
         open(path_prefix + '.top{}'.format(k) + '.{}'.format(trg), 'w') as trg_file:
            for pair in scored_data[:k]:
                src_file.write(pair[0])
                trg_file.write(pair[1])

def write_random_k_to_disc(data, path_prefix, src, trg, k):
    sampled_data = random.sample(data, min(k, len(data)))
    with open(path_prefix + '.random{}'.format(k) + '.{}'.format(src), 'w') as src_file, \
         open(path_prefix + '.random{}'.format(k) + '.{}'.format(trg), 'w') as trg_file:
            for pair in sampled_data[:k]:
                src_file.write(pair[0])
                trg_file.write(pair[1])

def load_general_data(raw_src, raw_tgt, vector_rep):
  loaded_vecs = np.load(vector_rep)
  logger.info("loaded {}".format(vector_rep))
  loaded_src_sents = open(raw_src).readlines()
  logger.info("loaded {}".format(raw_src))
  loaded_trg_sents = open(raw_tgt).readlines()
  logger.info("loaded {}".format(raw_tgt))
  pair_vecs = list(zip(loaded_src_sents, loaded_trg_sents, loaded_vecs))
  logger.info('loaded {} sentences'.format(len(pair_vecs)))
  return pair_vecs

def load_centriod(path):
  loaded_vecs = np.load(path)
  logger.info("loaded {}".format(loaded_vecs))
  return loaded_vecs


def score(data, centroid):
  # compute the distance from the centroid for each sample
  logger.info('computing scores...')
  scored = []
  for pair_vec_domain in tqdm(data, desc='scoring...'):
    vector_rep = pair_vec_domain[2]
    pair_vec_domain_score = (pair_vec_domain[0],
                             pair_vec_domain[1],
                             pair_vec_domain[2],
                             cosine_similarity(centroid.reshape(1, -1), vector_rep.reshape(1, -1))[0][0])
    scored.append(pair_vec_domain_score)
  logger.info('done.')

  # sort by score
  logger.info('ranking...')
  scored.sort(key=operator.itemgetter(3), reverse=True)
  logger.info('done.')

  # print first 5 and last 5
  logger.info("first:\n")
  for i in range(5):
    logger.info(scored[i][0])
  logger.info("last:\n")
  for i in range(1, 10):
    logger.info(scored[-i][0])
  return scored


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--input_dir", required=True, help="input file to compute domain vector")
  parser.add_argument("--src_lang", required=True, help="Source lang")
  parser.add_argument("--tgt_lang", default="en", help="Target lang")
  parser.add_argument("--input_file_prefix", required=True, help="input file prefix")
  parser.add_argument("--vector_rep", required=True, help="vector representation file")
  parser.add_argument("--centroid", required=True, help="centroid vector files")
  parser.add_argument("--output_dir", required=True, help="output file name")
  parser.add_argument('--k', help="comma separator list of k", default="50000,100000,500000")
  args = parser.parse_args()


  raw_src = "{}/{}.{}".format(args.input_dir, args.input_file_prefix, args.src_lang)
  raw_tgt = "{}/{}.{}".format(args.input_dir, args.input_file_prefix, args.tgt_lang)
  vector_rep = args.vector_rep
  general_data = load_general_data(raw_src, raw_tgt, vector_rep)
  centriod = load_centriod(args.centroid)
  scored = score(general_data, centriod)

  ks = map(int,args.k.split(','))
  output_prefix="{}/{}".format(args.output_dir, "news")
  for k in ks:
    write_topk_to_disc(scored, output_prefix, args.src_lang, args.tgt_lang, k)
    write_random_k_to_disc(scored, output_prefix, args.src_lang, args.tgt_lang, k)

if __name__ == "__main__":
  main()