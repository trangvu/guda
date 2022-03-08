#!/bin/bash

## Monarch env
#ALNMT_DIR=/home/xvuthith/da33/trang/uda-nmt/alnmt
#module load python/3.7.3-system
#module load cuda/10.1
#module load cudnn/7.6.5-cuda10.1
#source /home/xvuthith/da33/trang/uda-nmt/env/bin/activate

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

## Open Stack env
ALNMT_DIR='/home/ubuntu/projects/alnmt'
ROOT_DIR=$ALNMT_DIR
source /home/ubuntu/projects/guda-env/bin/activate

domain=$1
TGT_LANG=$2
SRC_LANG=en
lang_pair=en_${TGT_LANG}
DATE=`date '+%Y%m%d-%H%M%S'`
USR_DIR=$ALNMT_DIR/nmt/acl2021/cidds
seed=23
SOURCE_MODEL=$3
SOURCE_MODEL_CKPT=$ALNMT_DIR/models/$SOURCE_MODEL/checkpoint_best.pt

exp_name=from-en

DATASET=${domain}_${lang_pair}

DATA_DIR=$ALNMT_DIR/nmt/acl2021/preprocess/data-bin/ablation-uda-pseudo/$DATASET/$exp_name
SAVE_DIR=$ALNMT_DIR/models/ablation-uda-pseudo/$DATASET/from-en-pseudo-bi
echo '**************************************'
echo 'Train NMT '$SRC_LANG'-'$TGT_LANG' on domain '$domain
echo 'SRC_LANG          : '$SRC_LANG
echo 'TGT_LANG          : '$TGT_LANG
echo 'DATA_DIR          : '$DATA_DIR
echo 'SAVE_DIR          : '$SAVE_DIR
echo 'SOURCE_MODEL      : '$SOURCE_MODEL
echo 'SOURCE_MODEL_CKPT : '$SOURCE_MODEL_CKPT
echo 'EXP_NAME          : '${exp_name}
echo '**************************************'

set -x
fairseq-train $DATA_DIR \
  --task da_translation \
  --user-dir $USR_DIR \
  --criterion joint_domain_adapt_xent_with_smoothing \
  --max-epoch 50 --fp16 --patience 5 \
  --save-dir $SAVE_DIR \
  --source-lang $SRC_LANG --target-lang $TGT_LANG \
  --arch transformer_iwslt_de_en --share-decoder-input-output-embed \
  --optimizer adam --adam-betas "(0.9, 0.98)" --clip-norm 0.0 \
  --lr 1e-7 --lr-scheduler inverse_sqrt \
    --warmup-init-lr 1e-07 --warmup-updates 4000 --min-lr 1e-09 \
  --dropout 0.3 --weight-decay 0.0001 \
  --ddp-backend=no_c10d \
  --no-epoch-checkpoints \
  --discriminator-activation-fn gelu \
  --label-subset label --num-workers 4\
  --max-tokens 4096 --update-freq 1 \
  --tensorboard-logdir $SAVE_DIR/log --disable-src-loss --disable-tgt-loss

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
fairseq-score --sys $SAVE_DIR/test_best.detok.sys --ref $SAVE_DIR/test_best.detok.ref --sacrebleu | tee $SAVE_DIR/score.log
cat $SAVE_DIR/test_best.detok.sys | sacrebleu -w 2 $SAVE_DIR/test_best.detok.ref | tee $SAVE_DIR/score_best.log

