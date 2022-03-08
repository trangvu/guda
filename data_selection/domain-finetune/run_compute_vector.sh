#!/bin/bash

ALNMT_DIR='/home/trangvu/uda-nmt/alnmt'
ROOT_DIR=$ALNMT_DIR
module load python37
module load cuda-11.2.0-gcc-10.2.0-gsjevs3
source /home/trangvu/uda-nmt/env/bin/activate

language=$1
domain=$2

for l in $language en; do
    for f in train valid; do
        INPUT=$ALNMT_DIR/nmt/acl2021/preprocess/${domain}_en_${language}/$f.$l
        OUTPUT=$ALNMT_DIR/nmt/acl2021/preprocess/selection/${domain}_en_${language}/$f.$l.avg_pooled_vec.npy

        python3 compute_domain_vector.py --input $INPUT --output $OUTPUT --sentencepiece
    done
done