#!/bin/bash

LANGUAGES=( fr de cs ar )
SRC_DATASETS=( "wmt14_en_fr" "wmt20_en_de" "wmt20_en_cs" "news_en_ar")


ALNMT_DIR='/home/trangvu/uda-nmt/alnmt'
ROOT_DIR=$ALNMT_DIR
module load python37
module load cuda-11.2.0-gcc-10.2.0-gsjevs3
source /home/trangvu/uda-nmt/env/bin/activate

SUBSAMPLE_SIZE=2000000

for ((i=0;i<${#LANGUAGES[@]};++i)); do
    L=${LANGUAGES[i]}
    LANG_PAIR=${L}-en
    src_dataset=${SRC_DATASETS[i]}
    DATA_DIR=$ALNMT_DIR/nmt/acl2021/preprocess/$src_dataset
    echo "Subsample training data of ${src_dataset} for language pairs ${LANG_PAIR}"

    paste $DATA_DIR/train.$L $DATA_DIR/train.en > $DATA_DIR/corpus.$LANG_PAIR

    sample_corpus=corpus.subsample.$SUBSAMPLE_SIZE.$LANG_PAIR
    shuf -n $SUBSAMPLE_SIZE $DATA_DIR/corpus.$LANG_PAIR > $DATA_DIR/$sample_corpus

    cat $DATA_DIR/$sample_corpus | cut -f1  > $DATA_DIR/train.subsample.$L
    cat $DATA_DIR/$sample_corpus | cut -f2  > $DATA_DIR/train.subsample.en
done
