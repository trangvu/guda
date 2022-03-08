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

src=en
tgt=$1
lang_pair=en-$tgt
domain=$2
k=$3
MODEL_NAME=${lang_pair}-k.${k}

TOTAL_NUM_UPDATES=50000
MAX_SENTENCES=128
HEAD_NAME=domain_disc
NUM_CLASSES=2
LR=3e-05
WARMUP_UPDATES=400
USR_DIR=$ALNMT_DIR/nmt/acl2021/cidds
datadir=$ALNMT_DIR/nmt/acl2021/preprocess/data-bin/domain_disc/$lang_pair
savedir=$ALNMT_DIR/nmt/acl2021/models/domain_disc_cons/$lang_pair-k.${k}

mkdir -p $savedir

PRETRAINED_PATH=$ALNMT_DIR/nmt/acl2021/models/contrastive/$MODEL_NAME/checkpoint_last.pt
ARCH=contrastive_distill_bert
TASK=sentence_prediction_from_contrastive_bert
CRITERION=sentence_prediction_with_adaptive

l=$tgt
databin=$datadir/$domain/$l
echo "Train  domain classifier on English - constrastive $k cluster for ${lang_pair}"
echo '**************************************'
echo 'DATA_DIR  : '$databin
echo 'SAVE_DIR  : '$savedir
echo 'MODEL_NAME: '$MODEL_NAME
echo 'MODEL_DIR : '$PRETRAINED_PATH
echo 'OUTPUT    : '$savedir/$domain/$l
echo '**************************************'

echo "Train domain disc for ${lang_pair} for $domain - $l side"
fairseq-train $databin \
--pretrained-path $PRETRAINED_PATH \
--save-dir $savedir/$domain/$l \
--user-dir $USR_DIR \
--max-positions 512 \
--max-sentences $MAX_SENTENCES \
--arch $ARCH  --task $TASK \
--sentence-avg \
--criterion $CRITERION \
--classification-head-name $HEAD_NAME \
--num-classes $NUM_CLASSES \
--dropout 0.1 --attention-dropout 0.1 \
--weight-decay 0.1 --optimizer adam --adam-betas "(0.9, 0.98)" --adam-eps 1e-06 \
--clip-norm 0.0 \
--lr-scheduler polynomial_decay --lr $LR --total-num-update $TOTAL_NUM_UPDATES --warmup-updates $WARMUP_UPDATES \
--max-epoch 20 \
--best-checkpoint-metric accuracy --maximize-best-checkpoint-metric \
--shorten-method "truncate" \
--update-freq 1 --no-epoch-checkpoints \
--tensorboard-logdir $savedir/$domain/$l/log