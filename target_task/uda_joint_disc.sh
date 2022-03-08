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
module load python/3.7.2-gcc6
module load cuda/10.1
module load cudnn/7.6.5-cuda10.1
#source /home/xvuthith/da33_scratch/trang/newenv/bin/activate

domain=$1
SRC_LANG=$2
TGT_LANG=$3
lang_pair=$4
DATE=`date '+%Y%m%d-%H%M%S'`
USR_DIR=$ALNMT_DIR/nmt/acl2021/cidds
seed=23
SOURCE_MODEL=$5
SOURCE_MODEL_CKPT=$ALNMT_DIR/models/$SOURCE_MODEL/checkpoint_best.pt

seltype=$6

DATASET=${domain}_${lang_pair}

for L in $SRC_LANG $TGT_LANG; do
  DATA_DIR=$ALNMT_DIR/nmt/acl2021/preprocess/data-bin/uda/train/$lang_pair/${domain}/${seltype}/$L
  SAVE_DIR=$ALNMT_DIR/models/uda/${lang_pair}/$domain-${seltype}-$L
  echo '**************************************'
  echo 'Train NMT '$SRC_LANG'-'$TGT_LANG' on domain '$domain
  echo 'SRC_LANG          : '$SRC_LANG
  echo 'TGT_LANG          : '$TGT_LANG
  echo 'DATA_DIR          : '$DATA_DIR
  echo 'SAVE_DIR          : '$SAVE_DIR
  echo 'SOURCE_MODEL      : '$SOURCE_MODEL
  echo 'SOURCE_MODEL_CKPT : '$SOURCE_MODEL_CKPT
  echo 'DATA SELECTI      : '${seltype}
  echo '**************************************'


  fairseq-train $DATA_DIR \
      --pretrained-nmt $SOURCE_MODEL_CKPT \
      --task da_translation \
      --user-dir $USR_DIR \
      --criterion joint_domain_adapt_xent_with_smoothing \
      --max-epoch 20 --fp16 \
      --save-dir $SAVE_DIR \
      --source-lang $SRC_LANG --target-lang $TGT_LANG \
      --arch transformer_iwslt_de_en --share-decoder-input-output-embed \
      --optimizer adam --adam-betas "(0.9, 0.98)" --clip-norm 0.0 \
      --lr 5e-4 --lr-scheduler inverse_sqrt \
      --dropout 0.3 --weight-decay 0.0001 \
      --update-freq 16 \
      --ddp-backend=no_c10d \
      --patience 5 --no-epoch-checkpoints \
      --discriminator-activation-fn gelu \
      --label-subset label \
      --max-sentences 16 --sentence-avg --log-interval=100 \
      --tensorboard-logdir $SAVE_DIR/log

  echo '**************************************'
  echo 'Evaluate '$SRC_LANG'-'$TGT_LANG' ON '$DATASET
  echo 'SRC_LANG  : '$SRC_LANG
  echo 'TGT_LANG  : '$TGT_LANG
  echo 'DATA_DIR  : '$DATA_DIR
  echo 'SAVE_DIR  : '$SAVE_DIR
  echo '**************************************'

  EVAL_DIR=$ALNMT_DIR/nmt/acl2021/preprocess/data-bin/$DATASET
  fairseq-generate $EVAL_DIR \
      --path $SAVE_DIR/checkpoint_best.pt \
      --task da_translation \
      --user-dir $USR_DIR \
      --source-lang $SRC_LANG --target-lang $TGT_LANG \
      --beam 5 --remove-bpe=sentencepiece --lenpen 0.6 | tee $SAVE_DIR/test_best.log

  grep ^H $SAVE_DIR/test_best.log | cut -f3 | sacremoses detokenize > $SAVE_DIR/test_best.detok.sys
  grep ^T $SAVE_DIR/test_best.log | cut -f2 | sacremoses detokenize > $SAVE_DIR/test_best.detok.ref
  fairseq-score --sys $SAVE_DIR/test_best.detok.sys --ref $SAVE_DIR/test_best.detok.ref --sacrebleu | tee $SAVE_DIR/score_best.log

done