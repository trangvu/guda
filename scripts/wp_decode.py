#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import absolute_import, division, print_function, unicode_literals

import argparse

from tokenizers import  BertWordPieceTokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vocab", required=True,
                        help="vocab file")
    parser.add_argument("--lower_case", action='store_true', help="lower case")
    parser.add_argument("--input", required=True, help="input file to decode")
    parser.add_argument("--input_format", choices=["piece", "id"], default="piece")
    args = parser.parse_args()

    tokenizer = BertWordPieceTokenizer(args.vocab, lowercase=args.lower_case)

    if args.input_format == "piece":
        def decode(l):
            ids = [tokenizer.token_to_id(tok) for tok in l]
            return tokenizer.decode(l)
    elif args.input_format == "id":
        def decode(l):
            return tokenizer.decode(l)
    else:
        raise NotImplementedError

    def tok2int(tok):
        # remap reference-side <unk> (represented as <<unk>>) to 0
        return int(tok) if tok != "<<unk>>" else 0

    with open(args.input, "r", encoding="utf-8") as h:
        for line in h:
            if args.input_format == "id":
                print(decode(list(map(tok2int, line.rstrip().split()))))
            elif args.input_format == "piece":
                print(decode(line.rstrip().split()))

if __name__ == "__main__":
    main()
