#!/bin/bash

## Monarch env
ALNMT_DIR=/home/xvuthith/da33/trang/uda-nmt/alnmt
module load python/3.7.3-system
module load cuda/10.1
module load cudnn/7.6.5-cuda10.1
source /home/xvuthith/da33/trang/uda-nmt/env/bin/activate

## Fitcluster env
#ALNMT_DIR='/home/trangvu/uda-nmt/alnmt'
#ROOT_DIR=$ALNMT_DIR
#module load python37
#module load cuda-11.2.0-gcc-10.2.0-gsjevs3
#source /home/trangvu/uda-nmt/env/bin/activate

## m3 env
#ALNMT_DIR='/home/xvuthith/da33_scratch/trang/uda-nmt/alnmt'
#ROOT_DIR=$ALNMT_DIR
#module load python/3.7.2-gcc6
#module load cuda/10.1
#module load cudnn/7.6.5-cuda10.1
#source /home/xvuthith/da33_scratch/trang/newenv/bin/activate

src=en
tgt=$1
domain=$2
langpair=$src-$tgt
set -x
SRC_DIR=$ALNMT_DIR/nmt/acl2021/preprocess/selection/$langpair/news
TGT_DIR=$ALNMT_DIR/nmt/acl2021/preprocess/selection/$langpair/$domain
FILE_PATTERN=score.test.${langpair}.${src}.txt

python3 calculate_ced_and_ranking.py \
  --source-domain $SRC_DIR \
  --target-domain $TGT_DIR \
  --output-dir $TGT_DIR/sorted \
  --file-pattern $FILE_PATTERN \
  --k 500000
