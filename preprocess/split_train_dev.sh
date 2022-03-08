#!/bin/bash

echo "Running on M3"
ALNMT_DIR='/home/xvuthith/da33_scratch/trang/uda-nmt/alnmt'
ROOT_DIR=$ALNMT_DIR
module load python/3.7.2-gcc6
module load cuda/10.1
module load cudnn/7.6.5-cuda10.1
source /home/xvuthith/da33_scratch/trang/newenv/bin/activate
TEXT=$1
SRC_LANG=$2
TGT_LANG=$3
OUTDIR=data-bin/${TEXT}
prep=$TEXT
tmp=$prep/tmp
SP_MODEL_NAME=sp_${SRC_LANG}_${TGT_LANG}_unigram
VOCAB_FILE=${SP_MODEL_NAME}.vocab.txt

#for L in $SRC_LANG $TGT_LANG; do
    for f in train dev; do
        echo "encode sentencepiece to ${f}..."
        python3 ../scripts/spm_encode.py --model=${SP_MODEL_NAME}.model --inputs $tmp/$f.$SRC_LANG $tmp/$f.$TGT_LANG \
            --outputs $tmp/sp.$f.$SRC_LANG $tmp/sp.$f.$TGT_LANG --min-len 5 --max-len 1024
    done
#done
f=test
python3 ../scripts/spm_encode.py --model=${SP_MODEL_NAME}.model --inputs $tmp/$f.$SRC_LANG $tmp/$f.$TGT_LANG \
        --outputs $tmp/sp.$f.$SRC_LANG $tmp/sp.$f.$TGT_LANG

for L in $SRC_LANG $TGT_LANG; do
    cp $tmp/sp.test.$L $prep/test.$L
    cp $tmp/sp.train.$L $prep/train.$L
    cp $tmp/sp.dev.$L $prep/valid.$L | tee .$L
done

echo '**************************************'
echo 'BINARY NMT DATASET '$TEXT
echo 'SRC_LANG  : '$SRC_LANG
echo 'TGT_LANG  : '$TGT_LANG
echo 'DATA_DIR  : '$TEXT
echo 'SAVE_DIR  : '$OUTDIR
echo 'VOCAB_FILE: '$VOCAB_FILE
echo '**************************************'

fairseq-preprocess --source-lang $SRC_LANG --target-lang $TGT_LANG \
    --trainpref $TEXT/train --validpref $TEXT/valid --testpref $TEXT/test \
    --destdir $OUTDIR \
    --srcdict $VOCAB_FILE --tgtdict $VOCAB_FILE \
    --workers 20