#!/bin/bash

ALNMT_DIR='/home/trangvu/uda-nmt/alnmt'
ROOT_DIR=$ALNMT_DIR
module load python37
module load cuda-11.2.0-gcc-10.2.0-gsjevs3
source /home/trangvu/uda-nmt/env/bin/activate

K=5
DOMAIN=$1
DATA_DIR=$ALNMT_DIR/nmt/acl2021/preprocess/cons_selection/en-de-k.${K}/$DOMAIN/en
INPUT=$DATA_DIR/merge.txt
SRC_DIR=$ALNMT_DIR/nmt/acl2021/data_selection/contrastive/ablation-selection-threshold

OUTDIR=$ALNMT_DIR/nmt/acl2021/preprocess/threshold/$DOMAIN
mkdir -p $OUTDIR
cd $SRC_DIR && python3 split-by-threshold.py \
        --input $INPUT \
        --outdir $OUTDIR

cd $OUTDIR && rename txt de-en.de *
cp $ALNMT_DIR/nmt/acl2021/preprocess/data-bin/wmt20_en_de/dict.* $OUTDIR