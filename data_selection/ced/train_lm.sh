#!/bin/bash

ALNMT_DIR='/home/trangvu/uda-nmt/alnmt'
ROOT_DIR=$ALNMT_DIR
module load python37
module load cuda-11.2.0-gcc-10.2.0-gsjevs3
source /home/trangvu/uda-nmt/env/bin/activate

DATASET=$1
prefix="-"$2
DATE=`date '+%Y%m%d-%H%M%S'`
DATA_DIR=$ALNMT_DIR/nmt/acl2021/preprocess/data-bin/$DATASET
SAVE_DIR=$ALNMT_DIR/models/$DATASET$prefix
seed=23
echo '**************************************'
echo 'Train language model on dataset '$DATASET
echo 'DATA_DIR  : '$DATA_DIR
echo 'SAVE_DIR  : '$SAVE_DIR
echo '**************************************'


fairseq-train $DATA_DIR --task language_modeling \
    --save-dir $SAVE_DIR \
    --arch transformer_lm --share-decoder-input-output-embed \
     --dropout 0.1 \
      --optimizer adam --adam-betas '(0.9, 0.98)' --weight-decay 0.01 --clip-norm 0.0 \
      --lr 0.0005 --lr-scheduler inverse_sqrt --warmup-updates 4000 --warmup-init-lr 1e-07 \
      --tokens-per-sample 512 --sample-break-mode none \
      --max-tokens 4096 --update-freq 80 \
      --fp16 --seed $seed \
      --max-update 2000000 \
      --tensorboard-logdir $SAVE_DIR/log

