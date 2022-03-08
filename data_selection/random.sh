#!/bin/bash

ALNMT_DIR='/home/xvuthith/da33_scratch/trang/uda-nmt/alnmt'
ROOT_DIR=$ALNMT_DIR
module load python/3.7.2-gcc6
module load cuda/10.1
module load cudnn/7.6.5-cuda10.1
source $HOME/da33_scratch/trang/newenv/bin/activate

DATASET=$1
src=$2
tgt=en
OUTDIR=$ALNMT_DIR/nmt/acl2021/preprocess/selection/$DATASET/

mkdir -p $OUTDIR

SENT_SIZE=500000
TOKEN_SIZE=25000000

data_dir=$ALNMT_DIR/nmt/acl2021/preprocess/monolingual
tmp=$data_dir/rand_tmp
mkdir -p $tmp
for l in $src $tgt; do
    echo "Shuffle and select ${SENT_SIZE} sentences and ${TOKEN_SIZE} tokens"
    shuf -n $SENT_SIZE $data_dir/mono.10m.$l > $OUTDIR/random.$l
    echo
done
