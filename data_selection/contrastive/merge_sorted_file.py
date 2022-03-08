'''
Created by trangvu on 29/01/21
'''
import argparse
import glob
import heapq
import logging

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--input-dir", required=True, help="input dir")
  parser.add_argument("--output-file", required=True, help="output file name")
  parser.add_argument("--file-pattern", required=True, help="output file name")
  args = parser.parse_args()

  # Merge the sorted files
  # fnames = sorted(glob.glob(f"{output_dir}/sorted*"))
  # file_handlers = [open(os.path.join(output_dir, fnames), 'r') for f in fnames]

  def keyfunc(line):
      return line.split('\t')[2]

  def decorated_file(f, key):
      for line in f:
          yield (key(line), line)
  print(f"{args.input_dir}/{args.file_pattern}")
  filenames = glob.glob(f"{args.input_dir}/{args.file_pattern}")
  print("Found {} file with pattern {}".format(len(filenames), args.file_pattern))
  files = map(open, filenames)
  outfile = open(args.output_file, 'w')

  for line in heapq.merge(*[decorated_file(f, keyfunc) for f in files], reverse=True):
      outfile.write(line[1])

if __name__ == "__main__":
  main()
