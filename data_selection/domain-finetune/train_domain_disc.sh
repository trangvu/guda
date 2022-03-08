#!/bin/bash

ALNMT_DIR='/home/trangvu/uda-nmt/alnmt'
ROOT_DIR=$ALNMT_DIR
module load python37
module load cuda-11.2.0-gcc-10.2.0-gsjevs3
source /home/trangvu/uda-nmt/env/bin/activate

src=en
tgt=$1
lang_pair=en-$tgt
domain=$2

TOTAL_NUM_UPDATES=50000
MAX_SENTENCES=64
HEAD_NAME=domain_disc
NUM_CLASSES=2
LR=1e-05
WARMUP_UPDATES=400
USR_DIR=$ALNMT_DIR/nmt/acl2021/cidds
datadir=$ALNMT_DIR/nmt/acl2021/preprocess/data-bin/domain_disc/$lang_pair
savedir=$ALNMT_DIR/models/domain_disc/$lang_pair
for l in $src $tgt; do
  echo "Train domain disc for ${lang_pair} for $domain - $l side"
  databin=$datadir/$domain/$l
  fairseq-train $databin \
    --save-dir $savedir/$domain/$l \
    --user-dir $USR_DIR \
    --max-positions 512 \
    --max-sentences $MAX_SENTENCES \
    --arch hf_distill_bert --task sentence_prediction_from_pretrained_bert \
    --sentence-avg \
    --criterion sentence_prediction \
    --classification-head-name $HEAD_NAME \
    --num-classes $NUM_CLASSES \
    --dropout 0.1 --attention-dropout 0.1 \
    --weight-decay 0.1 --optimizer adam --adam-betas "(0.9, 0.98)" --adam-eps 1e-06 \
    --clip-norm 0.0 \
    --lr-scheduler polynomial_decay --lr $LR --total-num-update $TOTAL_NUM_UPDATES --warmup-updates $WARMUP_UPDATES \
    --max-epoch 20 \
    --best-checkpoint-metric accuracy --maximize-best-checkpoint-metric \
    --shorten-method "truncate" \
    --update-freq 8 --no-epoch-checkpoints
done