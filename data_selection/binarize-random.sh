#!/bin/bash

ALNMT_DIR='/home/trangvu/uda-nmt/alnmt'
ROOT_DIR=$ALNMT_DIR
module load python37
module load cuda-11.2.0-gcc-10.2.0-gsjevs3
source /home/trangvu/uda-nmt/env/bin/activate
LANGUAGES=( fr de cs )
DATASETS=( law ted med it koran )
tgt=en


for ((i=0;i<${#LANGUAGES[@]};++i)); do
    src=${LANGUAGES[i]}
    dataset=${DATASETS[i]}
    fairseq-preprocess --source-lang $src --target-lang $tgt \
    --trainpref $dataset/train \
    --destdir data-bin/wmt20_en_de \
    --srcdict sp_de_en_unigram.vocab.txt --tgtdict sp_de_en_unigram.vocab.txt \
    --workers 20
done









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

