#!/bin/bash

SRC_LANG=$1
TGT_LANG=$2
SP_MODEL_NAME=sp_${SRC_LANG}_${TGT_LANG}
VOCAB_FILE=${SP_MODEL_NAME}.vocab.txt
DOMAIN_NAME=$3

prep=${DOMAIN_NAME}_${SRC_LANG}_${TGT_LANG}
tmp=$prep/tmp
OUTDIR=data-bin/$prep

echo '**************************************'
echo 'PREPARE DATA FOR DOMAIN '$DOMAIN_NAME
echo 'SRC_LANG  : '$SRC_LANG
echo 'TGT_LANG  : '$TGT_LANG
echo 'VOCAB_FILE: '$VOCAB_FILE
echo '**************************************'

echo "Apply sentence piece"

for L in $src $tgt; do
    for f in train.$L valid.$L test.$L; do
        echo "encode sentencepiece to ${f}..."
        python3 ../scripts/spm_encode.py --model=${SP_MODEL_NAME}.model --inputs $tmp/$f --outputs $tmp/sp.$f
    done
done

for L in $SRC_LANG $TGT_LANG; do
    cp $tmp/sp.test.$L $prep/test.$L
    cp $tmp/sp.train.$L $prep/train.$L
    cp $tmp/sp.valid.$L $prep/valid.$L
done

echo "Run FairSeq preprocess"
echo '**************************************'
echo 'BINARY NMT DATASET '$prep
echo 'SRC_LANG  : '$SRC_LANG
echo 'TGT_LANG  : '$TGT_LANG
echo 'DATA_DIR  : '$prep
echo 'SAVE_DIR  : '$OUTDIR
echo 'VOCAB_FILE: '$VOCAB_FILE
echo '**************************************'

fairseq-preprocess --source-lang $SRC_LANG --target-lang $TGT_LANG \
    --trainpref $prep/train --validpref $prep/valid --testpref $prep/test \
    --destdir $OUTDIR \
    --srcdict $VOCAB_FILE --tgtdict $VOCAB_FILE \
    --workers 20