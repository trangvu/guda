#!/bin/bash

## prepare  binary data  for random & target first
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
lang_pair=en_$tgt
src_dataset=$2

bin_outdir=$ALNMT_DIR/nmt/acl2021/preprocess/data-bin/uda/$src_dataset
rm -r -f $bin_outdir
mkdir -p $bin_outdir

SP_MODEL_NAME=sp_${tgt}_${src}_unigram
if [[ $tgt == "ar" ]]; then
  SRC_VOCAB=$ALNMT_DIR/nmt/acl2021/preprocess/${SP_MODEL_NAME}-${src}.vocab.txt
  TGT_VOCAB=$ALNMT_DIR/nmt/acl2021/preprocess/${SP_MODEL_NAME}-${tgt}.vocab.txt
else
  SRC_VOCAB=$ALNMT_DIR/nmt/acl2021/preprocess/${SP_MODEL_NAME}.vocab.txt
  TGT_VOCAB=$ALNMT_DIR/nmt/acl2021/preprocess/${SP_MODEL_NAME}.vocab.txt
fi

# combine training data:
# create soft link
# from 3 sources:
# + src domain parallel
# + German tgt domain and its translation
# + English selection and its translation
# prepare corresponding labels
domain=ted
RAW_SRC_DIR=$ALNMT_DIR/nmt/acl2021/preprocess/selection/${domain}_en_$tgt

## Binarize the subset of source domain stored in  $ALNMT_DIR/nmt/acl2021/preprocess/selection
  valid_dir=$ALNMT_DIR/nmt/acl2021/preprocess/${domain}_en_$tgt
  f=train.$tgt-en
  fairseq-preprocess \
    --source-lang $src --target-lang $tgt \
    --srcdict $SRC_VOCAB \
    --tgtdict $TGT_VOCAB \
    --trainpref $RAW_SRC_DIR/$f \
    --validpref $valid_dir/valid \
    --destdir $bin_outdir \
    --workers 20
  for L in $src $tgt; do
    f=dict.$L.txt
    sed -i 's/^<unk> 1/<unk> 1 #fairseq:overwrite/g' $bin_outdir/$f
    sed -i 's/^<s> 1/<s> 1 #fairseq:overwrite/g' $bin_outdir/$f
    sed -i 's/^<\/s> 1/<\/s> 1 #fairseq:overwrite/g' $bin_outdir/$f
  done
TRAIN_SIZE=`wc -l $RAW_SRC_DIR/train.$tgt-en.$src | cut -d' ' -f1`
yes 0 | head -n $TRAIN_SIZE > $bin_outdir/train.label
