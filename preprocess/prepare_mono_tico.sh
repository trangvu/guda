#!/bin/bash

wdir=/home/vuth0001/workspace/acl2021/data/tico_mono/ar

# ar
f=mono.tico.ar
awk '!/<doc docid=/ && !/<\/doc>/' bbc.ara |  > $f
awk '!/<doc docid=/ && !/<\/doc>/' dw.ara |  >> $f
cat ar.txt >>$f

# fr
f=mono.tico.fr
awk '!/<doc docid=/ && !/<\/doc>/' bbc.fra |  > $f
awk '!/<doc docid=/ && !/<\/doc>/' dw.fra |  >> $f
awk '!/<doc docid=/ && !/<\/doc>/' voa.fra |  >> $f
cat fr.txt >>$f


# en
f=mono.tico.en
awk '!/<doc docid=/ && !/<\/doc>/' dw.eng |  > $f
cat *.txt >> $f

# tokenize en and fr
SCRIPTS=../mosesdecoder/scripts
TOKENIZER=$SCRIPTS/tokenizer/tokenizer.perl
NORM_PUNC=$SCRIPTS/tokenizer/normalize-punctuation.perl
REM_NON_PRINT_CHAR=$SCRIPTS/tokenizer/remove-non-printing-char.perl
LANG=fr
cat orig/mono.tico.$LANG | perl $NORM_PUNC $LANG | \
            perl $REM_NON_PRINT_CHAR | \
            perl $TOKENIZER -threads 8 -a -l $LANG >> tmp/mono.tok.$LANG

# preprocess ar
./detect_en_sentence.pl < orig/mono.tico.ar > orig/mono.tico.clean.ar
camel_arclean orig/mono.tico.clean.ar | camel_word_tokenize -o tmp/mono.tok.ar

# Apply sp and binarize

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

src=en
tgt=$1
langpair=${src}_${tgt}
SP_MODEL_NAME=sp_${tgt}_${src}_unigram
SP_DIR=$ALNMT_DIR/nmt/acl2021/preprocess
SPM_ENCODE=$ALNMT_DIR/nmt/acl2021/scripts/spm_encode.py

dataset=mono_tico
data_dir=$ALNMT_DIR/nmt/acl2021/preprocess/$dataset
outdir=$ALNMT_DIR/nmt/acl2021/preprocess/data-bin/$dataset-$tgt

# apply sentencepiece
if [[ $tgt == "ar" ]]; then
  SRC_VOCAB=$ALNMT_DIR/nmt/acl2021/preprocess/${SP_MODEL_NAME}-${src}.vocab.txt
  TGT_VOCAB=$ALNMT_DIR/nmt/acl2021/preprocess/${SP_MODEL_NAME}-${tgt}.vocab.txt
  for l in $src $tgt; do
      f=mono.tok
      python3 $SPM_ENCODE --model=$SP_DIR/${SP_MODEL_NAME}-${l}.model \
            --inputs $data_dir/$f.$l \
            --outputs $data_dir/$langpair/sp.$f.$l
  done
else
  SRC_VOCAB=$ALNMT_DIR/nmt/acl2021/preprocess/${SP_MODEL_NAME}.vocab.txt
  TGT_VOCAB=$ALNMT_DIR/nmt/acl2021/preprocess/${SP_MODEL_NAME}.vocab.txt
  f=mono.tok
  for l in $src $tgt; do
    python3 $SPM_ENCODE --model=$SP_DIR/${SP_MODEL_NAME}.model \
        --inputs $data_dir/$f.$l \
        --outputs $data_dir/$langpair/sp.$f.$l
  done
fi

# preprocess valid and test
fairseq-preprocess \
      --only-source \
      --source-lang $src --target-lang $tgt \
      --srcdict $SRC_VOCAB \
      --tgtdict $TGT_VOCAB \
      --trainpref $data_dir/$langpair/sp.mono.tok \
      --destdir $outdir \
      --workers 20

fairseq-preprocess \
      --only-source \
      --source-lang $tgt --target-lang $src \
      --srcdict $TGT_VOCAB \
      --tgtdict $SRC_VOCAB \
      --trainpref $data_dir/$langpair/sp.mono.tok \
      --destdir $outdir \
      --workers 20

for L in $src $tgt; do
    f=dict.$L.txt
    sed -i $outdir/input0/$f 's/^<unk> 1/<unk> 1 #fairseq:overwrite/g'
    sed -i $outdir/input0/$f 's/^<s> 1/<s> 1 #fairseq:overwrite/g'
    sed -i $outdir/input0/$f 's/^<\/s> 1/<\/s> 1 #fairseq:overwrite/g'
done


##########################
# Backtranslate
##########################