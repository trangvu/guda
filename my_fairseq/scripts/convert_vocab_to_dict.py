import argparse
import logging
import sys

logging.basicConfig(
    format='%(asctime)s | %(levelname)s | %(name)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    level=logging.INFO,
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input",
                        help="Input vocab file")
    parser.add_argument("--output-dir", required=True,
                        help="Output file")
    args = parser.parse_args()
    hf_pad = "[PAD]"
    hf_unk = "[UNK]"
    hf_eos = "[SEP]"
    hf_bos = "[CLS]"
    hf_mask = "[MASK]"
    pad = "<pad>"
    eos = "</s>"
    unk = "<unk>"
    bos = "<s>"
    mask = "<mask>"
    output_file = "{}/dict.txt".format(args.output_dir)
    with open(args.input, 'r') as fin, open(output_file, 'w') as fout:
        for line in fin:
            tok = line.strip()
            # if tok == hf_pad:
            #     tok = pad
            # elif tok == hf_unk:
            #     tok = unk
            # elif tok == hf_eos:
            #     tok = eos
            # elif tok == hf_bos:
            #     tok = bos
            # elif tok == hf_mask:
            #     tok = mask
            fout.write("{} {}\n".format(tok, 1))

if __name__ == "__main__":
  main()
