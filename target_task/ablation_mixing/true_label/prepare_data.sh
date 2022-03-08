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
domain=$2
src_dataset=$3
tgt_dataset=${domain}_${lang_pair}
num_tgt_label=$4

bin_outdir=$ALNMT_DIR/nmt/acl2021/preprocess/data-bin/ablation-uda/${domain}_${lang_pair}
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

SRC_DIR=$ALNMT_DIR/nmt/acl2021/preprocess/data-bin/uda/$src_dataset
TGT_DIR=$ALNMT_DIR/nmt/acl2021/preprocess/data-bin/$tgt_dataset

RAW_SRC_DIR=$ALNMT_DIR/nmt/acl2021/preprocess/selection/${domain}_en_$tgt
TGT_L1_DIR=$ALNMT_DIR/nmt/acl2021/preprocess/data-bin/beam/${domain}_en_${tgt}
RAW_TGT_L1_DIR=$ALNMT_DIR/nmt/acl2021/preprocess/${domain}_en_$tgt
TGT_EN_DIR=$ALNMT_DIR/nmt/acl2021/preprocess/data-bin/sel-beam/${domain}_en_${tgt}/$exp_name

echo "Prepare BI data"
  bi_dir=$bin_outdir/bi
  # source domain
  fin=train.en-$tgt
  fout=train.en-$tgt
  for ext in bin idx; do
    for L in en $tgt; do
       ln -s $SRC_DIR/${fin}.${L}.${ext} $bi_dir/${fout}.${L}.${ext}
    done
  done
  for L in en $tgt; do
       ln -s $SRC_DIR/dict.${L}.txt $bi_dir/dict.${L}.txt
  done
  # target domain
  fin=train.$tgt-en
  fout=train1.en-$tgt
  for ext in bin idx; do
    for L in en $tgt; do
       ln -s $TGT_DIR/${fin}.${L}.${ext} $bi_dir/${fout}.${L}.${ext}
    done
  done

  VALID_DIR=$ALNMT_DIR/ndmt/acl2021/preprocess/data-bin/${domain}_en_${tgt}
  for ext in bin idx; do
    for L in en $tgt; do
       ln -s $VALID_DIR/valid.${tgt}-en.${L}.${ext} $bi_dir/valid.en-${tgt}.${L}.${ext}
    done
  done

echo "Prepare BI+SRC+TGT"
bi_src_tgt_dir=$bin_outdir/bi_src_tgt
## Copy data by create softlink - source
  fin=train.en-$tgt
  fout=train.en-$tgt
  for ext in bin idx; do
    for L in en $tgt; do
       ln -s $SRC_DIR/${fin}.${L}.${ext} $bi_src_tgt_dir/${fout}.${L}.${ext}
    done
  done
  for L in en $tgt; do
       ln -s $SRC_DIR/dict.${L}.txt $bi_src_tgt_dir/dict.${L}.txt
  done

## Copy data by create softlink - $tgt
  fin=train.$tgt-en
  fout=train1.en-$tgt
  for ext in bin idx; do
    for L in en $tgt; do
        ln -s $TGT_DIR/${fin}.${L}.${ext} $bi_src_tgt_dir/${fout}.${L}.${ext}
    done
  done
## Copy data by create softlink - en
  fin=train.$tgt-en
  fout=train2.en-$tgt
  for ext in bin idx; do
    for L in en $tgt; do
        ln -s $TGT_DIR/${fin}.${L}.${ext} $bi_src_tgt_dir/${fout}.${L}.${ext}
    done
  done

## Copy validation dataset
  VALID_DIR=$ALNMT_DIR/ndmt/acl2021/preprocess/data-bin/${domain}_en_${tgt}
  for ext in bin idx; do
    for L in en $tgt; do
       ln -s $VALID_DIR/valid.${tgt}-en.${L}.${ext} $bi_src_tgt_dir/valid.en-${tgt}.${L}.${ext}
    done
  done

## Prepare labels
  cat $SRC_DIR/train.label > $bi_src_tgt_dir/train.label

  TRAIN_SIZE=`wc -l $RAW_TGT_L1_DIR/train.$src | cut -d' ' -f1`
  echo "TGT size "$TRAIN_SIZE
  yes 1 | head -n $num_tgt_label >> $bi_src_tgt_dir/train.label

  yes 1 | head -n $num_tgt_label >> $bi_src_tgt_dir/train.label


fairseq-preprocess \
  --only-source \
  --trainpref $bin_outdir/train.label \
  --destdir $bin_outdir/label \
  --workers 20
