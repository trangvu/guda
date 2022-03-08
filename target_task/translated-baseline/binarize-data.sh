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

## Note backtranslated data for each domain is saved at
## preprocess/selection/it_en_fr/backtranslate/bt_target_en.*
## bt_target_en means en is source language, fr is target language
## Output dir: data-bin/from-en/it_en_fr: input file is target_en
##             data-bin/to-en/it_en_fr: input file is target_fr

DOMAIN=$1
SRC_LANG=$2
TGT_LANG=en
DATASET=${DOMAIN}_en_${SRC_LANG}

IN_DIR=$ALNMT_DIR/nmt/acl2021/preprocess/selection/$DATASET/backtranslate
DATA_BIN=$ALNMT_DIR/nmt/acl2021/preprocess/data-bin

SP_MODEL_NAME=sp_${SRC_LANG}_${TGT_LANG}_unigram
if [[ $SRC_LANG == "ar" ]]; then
  SRC_VOCAB=$ALNMT_DIR/nmt/acl2021/preprocess/${SP_MODEL_NAME}-${SRC_LANG}.vocab.txt
  TGT_VOCAB=$ALNMT_DIR/nmt/acl2021/preprocess/${SP_MODEL_NAME}-${TGT_LANG}.vocab.txt
else
  SRC_VOCAB=$ALNMT_DIR/nmt/acl2021/preprocess/${SP_MODEL_NAME}.vocab.txt
  TGT_VOCAB=$ALNMT_DIR/nmt/acl2021/preprocess/${SP_MODEL_NAME}.vocab.txt
fi

############# FROM-EN
INPUT_FILE="bt_target_en"
OUTDIR="${DATA_BIN}/from-en/${DATASET}"

## valid and test dataset
prep=$ALNMT_DIR/nmt/acl2021/preprocess/${DATASET}
echo "Binarize ${DATASET} from-en input ${IN_DIR}/${INPUT_FILE} to output ${OUTDIR}"

echo "Run FairSeq preprocess"
echo '**************************************'
echo 'BINARY NMT DATASET '$DATASET
echo 'SRC_LANG  : '$TGT_LANG
echo 'TGT_LANG  : '$SRC_LANG
echo 'DATA_DIR  : '$IN_DIR
echo 'SAVE_DIR  : '$OUTDIR
echo 'VOCAB_FILE: '$SRC_VOCAB
echo '**************************************'

fairseq-preprocess --source-lang $TGT_LANG --target-lang $SRC_LANG \
    --trainpref $IN_DIR/$INPUT_FILE --validpref $prep/valid --testpref $prep/test \
    --destdir $OUTDIR \
    --srcdict $TGT_VOCAB --tgtdict $SRC_VOCAB \
    --workers 20

for L in $SRC_LANG $TGT_LANG; do
   f=dict.$L.txt
    sed -i 's/^<unk> 1/<unk> 1 #fairseq:overwrite/g' $OUTDIR/$f
    sed -i 's/^<s> 1/<s> 1 #fairseq:overwrite/g' $OUTDIR/$f
    sed -i 's/^<\/s> 1/<\/s> 1 #fairseq:overwrite/g' $OUTDIR/$f
done



############# TO-EN
INPUT_FILE="bt_target_${SRC_LANG}"
OUTDIR="${DATA_BIN}/to-en/${DATASET}"

## valid and test dataset
prep=$ALNMT_DIR/nmt/acl2021/preprocess/${DATASET}
echo "Binarize ${DATASET} from-en input ${IN_DIR}/${INPUT_FILE} to output ${OUTDIR}"

echo "Run FairSeq preprocess"
echo '**************************************'
echo 'BINARY NMT DATASET '$DATASET
echo 'SRC_LANG  : '$SRC_LANG
echo 'TGT_LANG  : '$TGT_LANG
echo 'DATA_DIR  : '$IN_DIR
echo 'SAVE_DIR  : '$OUTDIR
echo 'VOCAB_FILE: '$SRC_VOCAB
echo '**************************************'

fairseq-preprocess --source-lang $SRC_LANG --target-lang $TGT_LANG  \
    --trainpref $IN_DIR/$INPUT_FILE --validpref $prep/valid --testpref $prep/test \
    --destdir $OUTDIR \
    --srcdict $SRC_VOCAB --tgtdict $TGT_VOCAB \
    --workers 20

for L in $SRC_LANG $TGT_LANG; do
   f=dict.$L.txt
    sed -i 's/^<unk> 1/<unk> 1 #fairseq:overwrite/g' $OUTDIR/$f
    sed -i 's/^<s> 1/<s> 1 #fairseq:overwrite/g' $OUTDIR/$f
    sed -i 's/^<\/s> 1/<\/s> 1 #fairseq:overwrite/g' $OUTDIR/$f
done
