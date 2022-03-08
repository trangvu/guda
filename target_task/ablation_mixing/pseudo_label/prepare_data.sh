#!/bin/bash

## prepare  binary data  for random & target first
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

src=en
tgt=$1
lang_pair=en_$tgt
domain=$2
src_dataset=$3
tgt_dataset=${domain}_${lang_pair}
K=5
M=500000
exp_name=cons-k.${K}-m.${M}
TRAIN_1=$4
TRAIN_2=$5
bin_outdir=$ALNMT_DIR/nmt/acl2021/preprocess/data-bin/ablation-uda-pseudo/${domain}_${lang_pair}
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

# combine training data: - create 2 dataset for from-en and to-en
#    + train: src domain parallel
#    + train1: backtranslate new domain
#    + train2: forward-translate new domain
# create soft link
# from 3 sources:
# + src domain parallel
# + German tgt domain and its translation
# + English selection and its translation
# prepare corresponding labels

SRC_DIR=$ALNMT_DIR/nmt/acl2021/preprocess/data-bin/uda/$src_dataset


RAW_SRC_DIR=$ALNMT_DIR/nmt/acl2021/preprocess/selection/${domain}_en_$tgt
TGT_L1_DIR=$ALNMT_DIR/nmt/acl2021/preprocess/data-bin/beam/${domain}_en_${tgt}
RAW_TGT_L1_DIR=$ALNMT_DIR/nmt/acl2021/preprocess/${domain}_en_$tgt
TGT_EN_DIR=$ALNMT_DIR/nmt/acl2021/preprocess/data-bin/sel-beam/${domain}_en_${tgt}/$exp_name


# translate en-$tgt
echo "Prepare BI+TGT+SRC from-en"
  from_en_dir=$bin_outdir/from-en
  rm -r -f $from_en_dir
  mkdir -p $from_en_dir
  # source domain
  fin=train.en-$tgt
  fout=train.en-$tgt
  for ext in bin idx; do
    for L in en $tgt; do
       ln -s $SRC_DIR/${fin}.${L}.${ext} $from_en_dir/${fout}.${L}.${ext}
    done
  done
  for L in en $tgt; do
       ln -s $SRC_DIR/dict.${L}.txt $from_en_dir/dict.${L}.txt
  done

  # target domain - backtranslated data
  fin=train.en-$tgt
  fout=train1.en-$tgt
  for ext in bin idx; do
    for L in en $tgt; do
        ln -s $TGT_L1_DIR/${fin}.${L}.${ext} $from_en_dir/${fout}.${L}.${ext}
    done
  done

  ## target domain - forward-translated data
  fin=train.en-$tgt
  fout=train2.en-$tgt
  for ext in bin idx; do
    for L in en $tgt; do
        ln -s $TGT_EN_DIR/${fin}.${L}.${ext} $from_en_dir/${fout}.${L}.${ext}
    done
  done

  # validation data
  VALID_DIR=$ALNMT_DIR/nmt/acl2021/preprocess/data-bin/${domain}_en_${tgt}
  for ext in bin idx; do
    for L in en $tgt; do
       ln -s $VALID_DIR/valid.${tgt}-en.${L}.${ext} $from_en_dir/valid.en-${tgt}.${L}.${ext}
    done
  done

  ## Prepare labels
  #cat $SRC_DIR/train.label > $from_en_dir/train.label
  yes 0 | head -n 500000 > $from_en_dir/train.label

    #TRAIN_SIZE=`wc -l $RAW_TGT_L1_DIR/train.$src | cut -d' ' -f1`
      #echo "TGT size "$TRAIN_SIZE
  yes 1 | head -n $TRAIN_1 >> $from_en_dir/train.label
  yes 1 | head -n $TRAIN_2 >> $from_en_dir/train.label


  fairseq-preprocess \
  --only-source \
  --trainpref $from_en_dir/train.label \
  --destdir $from_en_dir/label \
  --workers 20


####################################################################
# translate $tgt-en
echo "Prepare BI+SRC+TGT - to-en"
to_en_dir=$bin_outdir/to-en
rm -r -f $to_en_dir
mkdir -p $to_en_dir
## Copy data by create softlink - source
  fin=train.en-$tgt
  fout=train.$tgt-en
  for ext in bin idx; do
    for L in en $tgt; do
       ln -s $SRC_DIR/${fin}.${L}.${ext} $to_en_dir/${fout}.${L}.${ext}
    done
  done
  for L in en $tgt; do
       ln -s $SRC_DIR/dict.${L}.txt $to_en_dir/dict.${L}.txt
  done

  ## target domain - backward-translated data
  fin=train.en-$tgt
  fout=train1.$tgt-en
  for ext in bin idx; do
    for L in en $tgt; do
        ln -s $TGT_EN_DIR/${fin}.${L}.${ext} $to_en_dir/${fout}.${L}.${ext}
    done
  done

  # target domain - forward-translated data
  fin=train.en-$tgt
  fout=train2.$tgt-en
  for ext in bin idx; do
    for L in en $tgt; do
        ln -s $TGT_L1_DIR/${fin}.${L}.${ext} $to_en_dir/${fout}.${L}.${ext}
    done
  done


## Copy validation dataset
  VALID_DIR=$ALNMT_DIR/nmt/acl2021/preprocess/data-bin/${domain}_en_${tgt}
  for ext in bin idx; do
    for L in en $tgt; do
       ln -s $VALID_DIR/valid.${tgt}-en.${L}.${ext} $to_en_dir/valid.${tgt}-en.${L}.${ext}
    done
  done

## Prepare labels
  yes 0 | head -n 500000 > $to_en_dir/train.label
  #TRAIN_SIZE=`wc -l $RAW_TGT_L1_DIR/train.$src | cut -d' ' -f1`
  #echo "TGT size "$TRAIN_SIZE
  yes 1 | head -n $TRAIN_2 >> $to_en_dir/train.label
  yes 1 | head -n $TRAIN_1 >> $to_en_dir/train.label


  fairseq-preprocess \
  --only-source \
  --trainpref $to_en_dir/train.label \
  --destdir $to_en_dir/label \
  --workers 20
