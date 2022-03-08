#!/bin/bash

ALNMT_DIR='/home/trangvu/uda-nmt/alnmt'
ROOT_DIR=$ALNMT_DIR
module load python37
module load cuda-11.2.0-gcc-10.2.0-gsjevs3
source /home/trangvu/uda-nmt/env/bin/activate


language=$1
dataset=$2

num_clusters=( 2 3 5 7 10 )

INPUT_DIR=$ALNMT_DIR/nmt/acl2021/preprocess/selection/$dataset
OUTPUT_DIR=$ALNMT_DIR/nmt/acl2021/preprocess/data-bin/$dataset

for L in $language en; do
    for k in "${num_clusters[@]}"; do
        f=train.subsample.$L.avg_pooled_vec.npy
        out=labels.$k.txt
        echo "Clustering $f into $k cluster"
        python3 clustering.py --data $INPUT_DIR/$f \
            --output $OUTPUT_DIR/$out \
            --k $k
    done
done