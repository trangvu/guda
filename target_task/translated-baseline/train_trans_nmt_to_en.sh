#!/bin/bash

echo "Running ${0} on M3"
ALNMT_DIR='/home/trangvu/uda-nmt/alnmt'
ROOT_DIR=$ALNMT_DIR
module load python37
module load cuda-11.2.0-gcc-10.2.0-gsjevs3
source /home/trangvu/uda-nmt/env/bin/activate
DATASET=$1
SRC_LANG=$2
TGT_LANG=en
prefix="-trans_beam_to_en"
DATE=`date '+%Y%m%d-%H%M%S'`
#DATA_DIR=$ALNMT_DIR/nmt/acl2021/preprocess/data-bin/to-en/$DATASET
DATA_DIR=$ALNMT_DIR/nmt/acl2021/preprocess/data-bin/beam/$DATASET
SAVE_DIR=$ALNMT_DIR/models/$DATASET$prefix-$DATE
seed=23
SOURCE_MODEL=$4
SOURCE_MODEL_CKPT=$ALNMT_DIR/models/$SOURCE_MODEL/checkpoint_best.pt
echo '**************************************'
echo 'Train NMT '$SRC_LANG'-'$TGT_LANG' on source dataset '$DATASET
echo 'SRC_LANG          : '$SRC_LANG
echo 'TGT_LANG          : '$TGT_LANG
echo 'DATA_DIR          : '$DATA_DIR
echo 'SAVE_DIR          : '$SAVE_DIR
echo 'SOURCE_MODEL      : '$SOURCE_MODEL
echo 'SOURCE_MODEL_CKPT : '$SOURCE_MODEL_CKPT
echo '**************************************'

fairseq-train $DATA_DIR \
    --restore-file $SOURCE_MODEL_CKPT \
    --reset-dataloader   --reset-dataloader --reset-meters --reset-optimizer \
    --max-epoch 30 --fp16 \
    --save-dir $SAVE_DIR \
    --source-lang $SRC_LANG --target-lang $TGT_LANG \
    --arch transformer_iwslt_de_en --share-decoder-input-output-embed \
    --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
    --lr 5e-4 --lr-scheduler inverse_sqrt --warmup-updates 100 \
    --dropout 0.3 --weight-decay 0.0001 \
    --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
    --max-tokens 4096 --update-freq 22 \
    --ddp-backend=no_c10d \
    --patience 5 --no-epoch-checkpoints \
    --tensorboard-logdir $SAVE_DIR/log

echo '**************************************'
echo 'Evaluate '$SRC_LANG'-'$TGT_LANG' ON '$DATASET
echo 'SRC_LANG  : '$SRC_LANG
echo 'TGT_LANG  : '$TGT_LANG
echo 'DATA_DIR  : '$DATA_DIR
echo 'SAVE_DIR  : '$SAVE_DIR
echo '**************************************'

fairseq-generate $DATA_DIR \
    --path $SAVE_DIR/checkpoint_best.pt \
    --source-lang $SRC_LANG --target-lang $TGT_LANG \
    --beam 5 --remove-bpe=sentencepiece --lenpen 0.6 | tee $SAVE_DIR/test.log

grep ^H $SAVE_DIR/test.log | cut -f3 | sacremoses detokenize > $SAVE_DIR/test.detok.sys
grep ^T $SAVE_DIR/test.log | cut -f2 | sacremoses detokenize > $SAVE_DIR/test.detok.ref
fairseq-score --sys $SAVE_DIR/test.detok.sys --ref $SAVE_DIR/test.detok.ref --sacrebleu | tee $SAVE_DIR/score.log