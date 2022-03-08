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
lang_pair=en_$tgt
selected_data=$2

orig_outdir=$ALNMT_DIR/nmt/acl2021/preprocess/data-bin/uda/orig/$lang_pair
bin_outdir=$ALNMT_DIR/nmt/acl2021/preprocess/data-bin/uda/train/$lang_pair
mkdir -p $orig_outdir $bin_outdir

SP_MODEL_NAME=sp_${tgt}_${src}_unigram
if [[ $tgt == "ar" ]]; then
  SRC_VOCAB=$ALNMT_DIR/nmt/acl2021/preprocess/${SP_MODEL_NAME}-${src}.vocab.txt
  TGT_VOCAB=$ALNMT_DIR/nmt/acl2021/preprocess/${SP_MODEL_NAME}-${tgt}.vocab.txt
else
  SRC_VOCAB=$ALNMT_DIR/nmt/acl2021/preprocess/${SP_MODEL_NAME}.vocab.txt
  TGT_VOCAB=$ALNMT_DIR/nmt/acl2021/preprocess/${SP_MODEL_NAME}.vocab.txt
fi

raw_indir=$ALNMT_DIR/nmt/acl2021/preprocess/selection
for domain in ted it koran law med; do
  data_dir=$raw_indir/${domain}_en_$tgt
  outdir=$orig_outdir/${domain}
  valid_dir=$ALNMT_DIR/nmt/acl2021/preprocess/${domain}_en_$tgt
  # binarize training data from source domain
  f=train.$tgt-en
  fairseq-preprocess \
    --source-lang $src --target-lang $tgt \
    --srcdict $SRC_VOCAB \
    --tgtdict $TGT_VOCAB \
    --trainpref $data_dir/$f \
    --validpref $valid_dir/valid \
    --destdir $outdir/news \
    --workers 20
  for L in $src $tgt; do
    f=dict.$L.txt
    sed -i 's/^<unk> 1/<unk> 1 #fairseq:overwrite/g' $outdir/news/$f
    sed -i 's/^<s> 1/<s> 1 #fairseq:overwrite/g' $outdir/news/$f
    sed -i 's/^<\/s> 1/<\/s> 1 #fairseq:overwrite/g' $outdir/news/$f
  done
  TRAIN_SIZE=`wc -l $data_dir/train.$tgt-en.$src | cut -d' ' -f1`
  yes 0 | head -n $TRAIN_SIZE > $outdir/news/train.label

  # binarize training data of groundtruth (ref)
  # we will create 2 dataset - one with true source side, and one with true target side
  # store in $outdir/target/en (en is true), and $outdir/target/$tgt ($tgt is true)
  data_dir=$raw_indir/${domain}_en_$tgt/backtranslate
  for seltype in target $selected_data; do
    for L in $src $tgt; do
      outdir=$orig_outdir/${domain}/${seltype}/${L}
      f=bt_${seltype}_${L}
      fairseq-preprocess \
      --source-lang $src --target-lang $tgt \
      --srcdict $SRC_VOCAB \
      --tgtdict $TGT_VOCAB \
      --trainpref $data_dir/$f \
      --destdir $outdir \
      --workers 20
      for t in $src $tgt; do
        f=dict.$t.txt
        sed -i 's/^<unk> 1/<unk> 1 #fairseq:overwrite/g' $outdir/$f
        sed -i 's/^<s> 1/<s> 1 #fairseq:overwrite/g' $outdir/$f
        sed -i 's/^<\/s> 1/<\/s> 1 #fairseq:overwrite/g' $outdir/$f
      done
      TRAIN_SIZE=`wc -l $data_dir/bt_${seltype}_${L}.$src | cut -d' ' -f1`
      yes 1 | head -n $TRAIN_SIZE > $outdir/train.label
    done
  done

  for seltype in $selected_data; do
    ## create training  data by copy valid data
    ## For each  domain and seltype, create 2 datasets according to indomain src or tgt lang
    domain_bindir=$bin_outdir/${domain}/${seltype}
    mkdir -p $domain_bindir/$src $domain_bindir/$tgt
    fromdir=$orig_outdir/${domain}
    for L in $src $tgt; do
      # indomain src
      cp $fromdir/news/dict.$L.txt $domain_bindir/$src
      cp $fromdir/news/train.$src-$tgt.$L.bin $domain_bindir/$src
      cp $fromdir/news/train.$src-$tgt.$L.idx $domain_bindir/$src
      cp $fromdir/news/valid.$src-$tgt.$L.bin $domain_bindir/$src
      cp $fromdir/news/valid.$src-$tgt.$L.idx $domain_bindir/$src
      cp $fromdir/target/${src}/train.$src-$tgt.$L.bin $domain_bindir/$src/train1.$src-$tgt.$L.bin
      cp $fromdir/target/${src}/train.$src-$tgt.$L.idx $domain_bindir/$src/train1.$src-$tgt.$L.idx
      cp $fromdir/${seltype}/${tgt}/train.$src-$tgt.$L.bin $domain_bindir/$src/train2.$src-$tgt.$L.bin
      cp $fromdir/${seltype}/${tgt}/train.$src-$tgt.$L.idx $domain_bindir/$src/train2.$src-$tgt.$L.idx

      # indomain tgt
      cp $fromdir/news/dict.$L.txt $domain_bindir/$tgt
      cp $fromdir/news/train.$src-$tgt.$L.bin $domain_bindir/$tgt
      cp $fromdir/news/train.$src-$tgt.$L.idx $domain_bindir/$tgt
      cp $fromdir/news/valid.$src-$tgt.$L.bin $domain_bindir/$tgt
      cp $fromdir/news/valid.$src-$tgt.$L.idx $domain_bindir/$tgt
      cp $fromdir/${seltype}/${src}/train.$src-$tgt.$L.bin $domain_bindir/$tgt/train1.$src-$tgt.$L.bin
      cp $fromdir/${seltype}/${src}/train.$src-$tgt.$L.idx $domain_bindir/$tgt/train1.$src-$tgt.$L.idx
      cp $fromdir/target/${tgt}/train.$src-$tgt.$L.bin $domain_bindir/$tgt/train2.$src-$tgt.$L.bin
      cp $fromdir/target/${tgt}/train.$src-$tgt.$L.idx $domain_bindir/$tgt/train2.$src-$tgt.$L.idx
    done

    cat $fromdir/news/train.label \
        $fromdir/target/${src}/train.label \
        $fromdir/${seltype}/${tgt}/train.label > $domain_bindir/$src/train.label
    fairseq-preprocess \
      --only-source \
      --trainpref $domain_bindir/$src/train.label \
      --destdir $domain_bindir/$src/label \
      --workers 20

    cat $fromdir/news/train.label \
        $fromdir/${seltype}/${src}/train.label \
        $fromdir/target/${tgt}/train.label > $domain_bindir/$tgt/train.label
    fairseq-preprocess \
      --only-source \
      --trainpref $domain_bindir/$tgt/train.label \
      --destdir $domain_bindir/$tgt/label \
      --workers 20
  done
 done

