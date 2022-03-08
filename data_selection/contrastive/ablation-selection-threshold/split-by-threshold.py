'''
Created by trangvu on 7/05/21
'''
import argparse
import logging
import fileinput

from tqdm import tqdm

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


def cli_main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--input', required=True, help="Scored input file")
  parser.add_argument('--outdir', required=True, help='Output dir')
  parser.add_argument('--out-filename', default="score_threshold", help='Output file name')
  args = parser.parse_args()

  out_prefix = "{}/{}".format(args.outdir, args.out_filename)
  infile = args.input

  logger.info("Split scored input {} by threshold".format(infile))


  threshold_cnt = []
  cur_threshold = 0.9
  cnt = 0
  num = int(cur_threshold * 10)
  filename = "{}.threshold-{}.txt".format(out_prefix, num)
  fout = open(filename, 'w')
  for line in tqdm(fileinput.input(infile)):
    data = line.strip().split('\t')
    score = float(data[1])
    sent = data[2]
    if score < cur_threshold:
      logger.info("Find {} sentences at threshold {}".format(cnt, cur_threshold))
      logger.info("Save it to {}".format(filename))
      fout.close()
      threshold_cnt.append((cur_threshold, cnt))
      cur_threshold -= 0.1
      cnt = 0
      num = int(cur_threshold * 10)
      filename = "{}.threshold-{}.txt".format(out_prefix, num)
      fout = open(filename, 'w')
    fout.write("{}\n".format(sent))
    cnt += 1

  logger.info("Finished spliting")
  logger.info("Histogram: {}".format(threshold_cnt))

if __name__ == '__main__':
  cli_main()