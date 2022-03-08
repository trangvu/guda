#!/bin/bash

CLUSTER=`hostname`
if [[ "$CLUSTER" == *"gadi"* ]]; then
  ALNMT_DIR='/home/565/tv2852/dz21/trang/alnmt'
  ROOT_DIR=$ALNMT_DIR
  echo "Running on NCI"
  module load python3/3.7.4
  module load cuda/10.1
  module load cudnn/7.6.5-cuda10.1
  source $ROOT_DIR/env/bin/activate
else
  if [[ "$CLUSTER" == *"m3"* ]]; then
    echo "Running on M3"
    ALNMT_DIR='/home/xvuthith/da33_scratch/trang/uda-nmt/alnmt'
    ROOT_DIR=$ALNMT_DIR
    module load python/3.7.2-gcc6
    module load cuda/10.1
    module load cudnn/7.6.5-cuda10.1
    source ~/da33_scratch/trang/newenv/bin/activate
    TEXT=$1
    SRC_LANG=$2
    TGT_LANG=$3
    OUTDIR=$4
    VOCAB_FILE=$5
  fi
fi

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

for L in $SRC_LANG $TGT_LANG; do
   f=dict.$L.txt
    sed -i $OUTDIR/$f 's/^<unk> 1/<unk> 1 #fairseq:overwrite/g'
    sed -i $OUTDIR/$f 's/^<s> 1/<s> 1 #fairseq:overwrite/g'
    sed -i $OUTDIR/$f 's/^<\/s> 1/<\/s> 1 #fairseq:overwrite/g'
done
