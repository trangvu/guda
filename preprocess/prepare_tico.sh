#!/bin/bash

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

dataset=tico19_${langpair}
data_dir=$ALNMT_DIR/nmt/acl2021/preprocess/$dataset
outdir=$ALNMT_DIR/nmt/acl2021/preprocess/data-bin/$dataset

# apply sentencepiece
if [[ $tgt == "ar" ]]; then
  SRC_VOCAB=$ALNMT_DIR/nmt/acl2021/preprocess/${SP_MODEL_NAME}-${src}.vocab.txt
  TGT_VOCAB=$ALNMT_DIR/nmt/acl2021/preprocess/${SP_MODEL_NAME}-${tgt}.vocab.txt
  for l in $src $tgt; do
    for f in valid test; do
      python3 $SPM_ENCODE --model=$SP_DIR/${SP_MODEL_NAME}-${l}.model \
            --inputs $data_dir/$f.$l \
            --outputs $data_dir/sp.$f.$l
    done
  done
else
  SRC_VOCAB=$ALNMT_DIR/nmt/acl2021/preprocess/${SP_MODEL_NAME}.vocab.txt
  TGT_VOCAB=$ALNMT_DIR/nmt/acl2021/preprocess/${SP_MODEL_NAME}.vocab.txt
  for f in valid test; do
      python3 $SPM_ENCODE --model=$SP_DIR/${SP_MODEL_NAME}.model \
            --inputs $data_dir/$f.$src $data_dir/$f.$tgt \
            --outputs $data_dir/sp.$f.$src $data_dir/sp.$f.$tgt
  done
fi

# preprocess valid and test
fairseq-preprocess \
      --source-lang $src --target-lang $tgt \
      --srcdict $SRC_VOCAB \
      --tgtdict $TGT_VOCAB \
      --validpref $data_dir/sp.valid \
      --testpref $data_dir/sp.test \
      --destdir $outdir \
      --workers 20

for L in $src $tgt; do
    f=dict.$L.txt
    sed -i $outdir/input0/$f 's/^<unk> 1/<unk> 1 #fairseq:overwrite/g'
    sed -i $outdir/input0/$f 's/^<s> 1/<s> 1 #fairseq:overwrite/g'
    sed -i $outdir/input0/$f 's/^<\/s> 1/<\/s> 1 #fairseq:overwrite/g'
done