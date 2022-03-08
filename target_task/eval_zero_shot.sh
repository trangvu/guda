#!/bin/bash

echo "Running on M3"
ALNMT_DIR='/home/xvuthith/da33_scratch/trang/uda-nmt/alnmt'
ROOT_DIR=$ALNMT_DIR
module load python/3.7.2-gcc6
module load cuda/10.1
module load cudnn/7.6.5-cuda10.1
source /home/xvuthith/da33_scratch/trang/newenv/bin/activate
DATASET=$1
SRC_LANG=$2
TGT_LANG=$3
SOURCE_MODEL=$4

DATE=`date '+%Y%m%d-%H%M%S'`
DATA_DIR=$ALNMT_DIR/nmt/acl2021/preprocess/data-bin/$DATASET
SAVE_DIR=$ALNMT_DIR/models/$DATASET-zeroshot-$DATE
MODEL_DIR=$ALNMT_DIR/models/$SOURCE_MODEL
seed=23
mkdir -p $SAVE_DIR
echo '**************************************'
echo 'Evaluate '$SRC_LANG'-'$TGT_LANG' ON '$DATASET
echo 'SRC_LANG     : '$SRC_LANG
echo 'TGT_LANG     : '$TGT_LANG
echo 'DATA_DIR     : '$DATA_DIR
echo 'SOURCE_MODEL : '$MODEL_DIR
echo 'SAVE_DIR     : '$SAVE_DIR
echo '**************************************'

fairseq-generate $DATA_DIR \
    --path $MODEL_DIR/checkpoint_best.pt \
    --source-lang $SRC_LANG --target-lang $TGT_LANG \
    --beam 5 --remove-bpe=sentencepiece --lenpen 0.6 | tee $SAVE_DIR/test.log

grep ^H $SAVE_DIR/test.log | cut -f3 | sacremoses detokenize > $SAVE_DIR/test.detok.sys
grep ^T $SAVE_DIR/test.log | cut -f2 | sacremoses detokenize > $SAVE_DIR/test.detok.ref
fairseq-score --sys $SAVE_DIR/test.detok.sys --ref $SAVE_DIR/test.detok.ref --sacrebleu | tee $SAVE_DIR/score.log