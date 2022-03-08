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
dataset=$2
langpair=$src-$tgt

echo "Backtranslate all monolingual data - keep in log file for later extraction"
dataset=mono_tico
databin=$ALNMT_DIR/nmt/acl2021/preprocess/data-bin/$dataset-$tgt
MODEL_DIR=$ALNMT_DIR/models
echo "Back translate from $tgt to en"

if [[ $tgt == "ar" ]]; then
  src_model=news_en_ar
else
  src_model=wmt14_en_fr
fi

f=train
# Backtranslate tgt -> en
  save_model_dir=${MODEL_DIR}/${src_model}
  data_dir=$databin
  outdir=$data_dir/bt.$tgt-en
  mkdir -p  $outdir
  fairseq-generate $data_dir \
        --path $save_model_dir/checkpoint_best.pt \
        --skip-invalid-size-inputs-valid-test \
        --max-tokens 32000 \
        --gen-subset $f \
        --source-lang $tgt --target-lang en \
        --sampling --beam 1  | tee  $outdir/$f.$tgt-en.bt.log

  echo "Extract BT data and save to "$outdir/bt_$f.$tgt-en
    python3 extract_bt_data.py \
    --minlen 1 --maxlen 250 --ratio 1.5 \
    --output $outdir/bt_$f.$tgt-en --srclang $tgt --tgtlang en \
    $outdir/$f.$tgt-en.bt.log



echo "Back translate from en to $tgt"
f=train

  save_model_dir=${MODEL_DIR}/${src_model}-from_en
  data_dir=$databin
  outdir=$data_dir/bt.en-$tgt
  mkdir -p  $outdir
  fairseq-generate $data_dir \
        --path $save_model_dir/checkpoint_best.pt \
        --skip-invalid-size-inputs-valid-test \
        --max-tokens 32000 \
        --gen-subset $f \
        --source-lang en --target-lang $tgt \
        --sampling --beam 1  | tee  $outdir/$f.en-$tgt.bt.log

  echo "Extract BT data and save to "$outdir/bt_$f.en-$tgt
    python3 extract_bt_data.py \
    --minlen 1 --maxlen 250 --ratio 1.5 \
    --output $outdir/bt_$f.en-$tgt --srclang $tgt --tgtlang en \
    $outdir/$f.en-$tgt.bt.log