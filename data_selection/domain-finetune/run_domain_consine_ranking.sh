#!/bin/bash

ALNMT_DIR='/home/trangvu/uda-nmt/alnmt'
ROOT_DIR=$ALNMT_DIR
module load python37
module load cuda-11.2.0-gcc-10.2.0-gsjevs3
source /home/trangvu/uda-nmt/env/bin/activate

language=$1
dataset=$2

DOMAINS=( law med ted it koran )

INPUT_DIR=$ALNMT_DIR/nmt/acl2021/preprocess/$dataset
VECTOR_REP=$ALNMT_DIR/nmt/acl2021/preprocess/selection/$dataset/train.subsample.en.avg_pooled_vec.npy

for domain in "${DOMAINS[@]}"; do
    CENTROID=$ALNMT_DIR/nmt/acl2021/preprocess/selection/${domain}_en_${language}/train.en.avg_pooled_vec.npy.centroid.npy
    OUTPUT_DIR=$ALNMT_DIR/nmt/acl2021/preprocess/selection/${domain}_en_${language}/src-ranking
    mkdir -p $OUTPUT_DIR
    python3 domain_cosine_ranking.py --input_dir $INPUT_DIR \
        --src_lang $language --tgt_lang en \
        --input_file_prefix train.subsample \
        --vector_rep $VECTOR_REP \
        --centroid $CENTROID \
        --output_dir $OUTPUT_DIR \
        --k 50000,500000,1000000,2000000
done