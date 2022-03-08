import argparse
import collections
import glob
import heapq
import multiprocessing
import os
import time
from operator import itemgetter
MoreLewisArgument = collections.namedtuple('MoreLewisArgument', ['source_dir', 'target_dir', 'file_pattern', 'num_workers', 'output_dir', 'k'])

class MoreLewisCalculator():
    def __init__(self, job_id, source_dir, target_dir, output_dir):
        self.job_id = job_id
        self.source_dir = source_dir
        self.target_dir = target_dir
        self.output_dir = output_dir


    def process(self, shard_id, source_file, target_file, outfile):
        score_dict = {}
        with open(os.path.join(self.source_dir, source_file), 'r') as fin:
            for line in fin:
                line=line.strip().split('\t')
                sample_id = line[0]
                score = float(line[1])
                sent = line[2]
                score_dict[sample_id] = (score, sent)
        print("Finish reading {} lines from source {}".format(len(score_dict), source_file))

        ced_scores = []
        with open(os.path.join(self.target_dir, target_file), 'r') as fin:
            for line in fin:
                line=line.strip().split('\t')
                sample_id = line[0]
                tgt_score = float(line[1])
                sent = line[2]
                src_score = score_dict[sample_id][0]
                src_sent = score_dict[sample_id][1]
                ced_score = tgt_score - src_score
                ced_scores.append((ced_score, sample_id, src_sent))
        ced_scores.sort(key=itemgetter(0), reverse=True)

        with open(os.path.join(self.output_dir, outfile), 'w') as fout:
            for (score, id, sent) in ced_scores:
                fout.write("{}\t{}\t{}\t{}\n".format(shard_id, id, score, sent))



    def finish(self):
        print("Job {} finished".format(self.job_id))

def sort_ced(job_id, args):
    def log(*args):
        msg = " ".join(map(str, args))
        print("Job {}:".format(job_id), msg)

    fnames = sorted(glob.glob(f"{args.source_dir}/shard*"))
    fnames = [f for (i, f) in enumerate(fnames)
              if i % args.num_workers == job_id]
    preprocessor = MoreLewisCalculator(job_id, args.source_dir, args.target_dir, args.output_dir)


    start_time = time.time()

    for file_no, fname in enumerate(fnames):
        basename = os.path.basename(fname)
        source_file= "{}/{}".format(fname, args.file_pattern)
        target_file = "{}/{}".format(fname, args.file_pattern)
        outfile_file = "sorted.{}.{}".format(basename, args.file_pattern)
        preprocessor.process(basename, source_file, target_file, outfile_file)

        elapsed = time.time() - start_time
        log("processed {:}/{:} files ({:.1f}%), ELAPSED: {:}s, ".format(
            file_no, len(fnames), 100.0 * file_no / len(fnames), int(elapsed)))
    preprocessor.finish()

def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--source-domain", required=True, help="input file to compute domain vector")
  parser.add_argument("--target-domain", required=True, help="output file name")
  parser.add_argument("--output-dir", required=True, help="output file name")
  parser.add_argument("--file-pattern", required=True, help="output file name")
  parser.add_argument("--k", help="Extract top k sentence")
  args = parser.parse_args()

  k = args.k
  src_path = args.source_domain
  tgt_path = args.target_domain
  file_pattern = args.file_pattern
  output_dir = args.output_dir
  shards = glob.glob(f"{src_path}/shard*")
  num_shard = len(shards)
  print("Found {} shards".format(num_shard))

  jobs = []
  MoreLewisArgument = collections.namedtuple('MoreLewisArgument',
                                             ['source_dir', 'target_dir', 'file_pattern', 'num_workers', 'output_dir',
                                              'k'])
  args = MoreLewisArgument(src_path, tgt_path, file_pattern, num_shard, output_dir, k)
  if num_shard == 1:
      sort_ced(0, args)
  else:
      for i in range(num_shard):
          job = multiprocessing.Process(target=sort_ced, args=(i, args))
          jobs.append(job)
          job.start()

      for job in jobs:
          job.join()

  # Merge the sorted files
  # fnames = sorted(glob.glob(f"{output_dir}/sorted*"))
  # file_handlers = [open(os.path.join(output_dir, fnames), 'r') for f in fnames]

  def keyfunc(line):
      return line.split('\t')[2]

  def decorated_file(f, key):
      for line in f:
          yield (key(line), line)

  filenames = glob.glob(f"{output_dir}/sorted*")
  files = map(open, filenames)
  outfile = open(os.path.join(output_dir,'merged.txt'), 'w')

  for line in heapq.merge(*[decorated_file(f, keyfunc) for f in files]):
      outfile.write(line[1])

if __name__ == "__main__":
  main()
