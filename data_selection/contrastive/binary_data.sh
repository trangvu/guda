#!/bin/bash

ALNMT_DIR='/home/trangvu/uda-nmt/alnmt'
ROOT_DIR=$ALNMT_DIR
module load python37
module load cuda-11.2.0-gcc-10.2.0-gsjevs3
source /home/trangvu/uda-nmt/env/bin/activate

src=en
tgt=$1
lang_pair=en-$tgt
dataset=$2
SP_DIR=$ALNMT_DIR/nmt/acl2021/preprocess
SPM_ENCODE=$ALNMT_DIR/nmt/acl2021/scripts/spm_encode.py
SP_MODEL_NAME=sp_${tgt}_en_unigram

WP_ENCODE=$ALNMT_DIR/nmt/acl2021/scripts/wp_encode.py
data_bin=$ALNMT_DIR/nmt/acl2021/preprocess/data-bin
outdir=$data_bin/contrastive/$lang_pair
data_dir=$ALNMT_DIR/nmt/acl2021/preprocess/$dataset
mkdir -p $outdir

SRC_VOCAB=distill_bert_vocab.txt
TGT_VOCAB=distill_bert_vocab.txt
VOCAB_FILE=${SP_MODEL_NAME}.vocab.txt
TEXT=$lang_pair
train_subset=train.subsample
echo "process $dataset for $lang_pair"

echo "Apply wordpiece"
python3 $WP_ENCODE --vocab vocab.txt \
      --inputs $data_dir/$train_subset.$src $data_dir/$train_subset.$tgt \
      --outputs $data_dir/wp.$train_subset.$src $data_dir/wp.$train_subset.$tgt

cat $data_dir/valid.$src | sed 's/ //g' | sed 's/▁/ /g' > $data_dir/valid.detok.$src
cat $data_dir/valid.$tgt | sed 's/ //g' | sed 's/▁/ /g' > $data_dir/valid.detok.$tgt

python3 $WP_ENCODE --vocab vocab.txt \
      --inputs $data_dir/valid.detok.$src $data_dir/valid.detok.$tgt \
      --outputs $data_dir/wp.valid.$src $data_dir/wp.valid.$tgt

echo "Preprocess"
fairseq-preprocess \
--source-lang $src --target-lang $tgt \
--srcdict $SRC_VOCAB \
--tgtdict $TGT_VOCAB \
--trainpref $data_dir/wp.$train_subset \
--validpref $data_dir/wp.valid \
--destdir $outdir/input0 \
--workers 20

for L in $src $tgt; do
    f=dict.$L.txt
    sed -i $outdir/input0/$f 's/^<unk> 1/<unk> 1 #fairseq:overwrite/g'
    sed -i $outdir/input0/$f 's/^<s> 1/<s> 1 #fairseq:overwrite/g'
    sed -i $outdir/input0/$f 's/^<\/s> 1/<\/s> 1 #fairseq:overwrite/g'
done

VALID_SIZE=`wc -l $data_dir/wp.valid.$src | cut -d' ' -f1 `
yes 0 | head -n $VALID_SIZE > $data_bin/$dataset/dummy.valid.txt
for k in 2 3 5 7 10; do
  echo "process labels $k"
  fairseq-preprocess \
    --only-source \
    --trainpref $data_bin/$dataset/labels.$k.txt \
    --validpref $data_bin/$dataset/dummy.valid.txt \
    --destdir $outdir/label$k/ \
    --workers 20
done