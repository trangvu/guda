#!/bin/bash

## Monarch env
#ALNMT_DIR=/home/xvuthith/da33/trang/uda-nmt/alnmt
#module load python/3.7.3-system
#module load cuda/10.1
#module load cudnn/7.6.5-cuda10.1
#source /home/xvuthith/da33/trang/uda-nmt/env/bin/activate

## Fitcluster env
ALNMT_DIR='/home/trangvu/uda-nmt/alnmt'
ROOT_DIR=$ALNMT_DIR
module load python37
module load cuda-11.2.0-gcc-10.2.0-gsjevs3
source /home/trangvu/uda-nmt/env/bin/activate

## m3 env
#ALNMT_DIR='/home/xvuthith/da33_scratch/trang/uda-nmt/alnmt'
#ROOT_DIR=$ALNMT_DIR
#module load python/3.7.2-gcc6
#module load cuda/10.1
#module load cudnn/7.6.5-cuda10.1
#source /home/xvuthith/da33_scratch/trang/newenv/bin/activate

language=$1
domain=$2
K=$3
dataset=${domain}_en_${language}
DATA_DIR=$ALNMT_DIR/nmt/acl2021/preprocess/selection/$dataset

INPUT_FILE=$ALNMT_DIR/nmt/acl2021/preprocess/cons_selection/en-${language}-k.${K}/${domain}/en/merge.txt
lang_pair=${language}-en
tmp=$DATA_DIR/tmp
mkdir -p $tmp

SPM_ENCODE=$ALNMT_DIR/nmt/acl2021/scripts/spm_encode.py
SP_MODEL_NAME=sp_${language}_en_unigram
SRC_VOCAB=$ALNMT_DIR/nmt/acl2021/preprocess/${SP_MODEL_NAME}.vocab.txt
TGT_VOCAB=$ALNMT_DIR/nmt/acl2021/preprocess/${SP_MODEL_NAME}.vocab.txt

for M in 10000 50000 100000 200000 500000 1000000; do
    f="cons-k.${K}-m.${M}.${lang_pair}.en"
    OUTFILE=$DATA_DIR/cons-k.${K}-m.${M}.${lang_pair}.en
    head -n $M $INPUT_FILE | cut -f3 > $tmp/$f
    # Apply sentence piece

  ### Apply sentence piece
  echo "encode sentencepiece to ${INPUT_FILE}..."
  python3 $SPM_ENCODE --model=$ALNMT_DIR/nmt/acl2021/preprocess/${SP_MODEL_NAME}.model --inputs $tmp/$f \
        --outputs $OUTFILE
done