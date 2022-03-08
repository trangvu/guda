#!/bin/bash

ALNMT_DIR='/home/trangvu/uda-nmt/alnmt'
ROOT_DIR=$ALNMT_DIR
module load python37
module load cuda-11.2.0-gcc-10.2.0-gsjevs3
source /home/trangvu/uda-nmt/env/bin/activate

language=$1
dataset=$2
for l in $language en; do
  f=train.subsample
  INPUT=$ALNMT_DIR/nmt/acl2021/preprocess/$dataset/$f.$l
  OUTPUT=$ALNMT_DIR/nmt/acl2021/preprocess/selection/$dataset/$f.$l.avg_pooled_vec.npy

  python3 compute_domain_vector.py --input $INPUT --output $OUTPUT --sentencepiece
done