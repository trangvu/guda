#!/bin/bash

ALNMT_DIR='/home/trangvu/uda-nmt/alnmt'
ROOT_DIR=$ALNMT_DIR
module load python37
module load cuda-11.2.0-gcc-10.2.0-gsjevs3
source /home/trangvu/uda-nmt/env/bin/activate

src=en
tgt=$1
src_dataset=$2
lang_pair=en-$tgt
raw_outdir=$ALNMT_DIR/nmt/acl2021/preprocess/domain_disc/$lang_pair
bin_outdir=$ALNMT_DIR/nmt/acl2021/preprocess/data-bin/domain_disc/$lang_pair
WP_VOCAB=$ALNMT_DIR/nmt/acl2021/data_selection/contrastive/vocab.txt
WP_ENCODE=$ALNMT_DIR/nmt/acl2021/scripts/wp_encode.py
SRC_VOCAB=$ALNMT_DIR/nmt/acl2021/data_selection/contrastive/distill_bert_vocab.txt

raw_news_dir=$ALNMT_DIR/nmt/acl2021/preprocess/$src_dataset

for domain in law it med ted koran; do
  news_indir=$ALNMT_DIR/nmt/acl2021/preprocess/selection/${domain}_en_${tgt}/src-ranking
  domain_indir=$ALNMT_DIR/nmt/acl2021/preprocess/${domain}_en_${tgt}
  raw_domain_dir=$ALNMT_DIR/nmt/acl2021/preprocess/${domain}_en_${tgt}
  mkdir -p $raw_outdir/$domain
  echo "====================================="
  echo "Preprocess $domain"
  echo "Preprocess news domain"

  #process train - negative
  f=negative
  cat $news_indir/$f.$src | sed 's/ //g' | sed 's/▁/ /g' > $raw_outdir/$domain/$f.$src
  cat $news_indir/$f.$tgt | sed 's/ //g' | sed 's/▁/ /g' > $raw_outdir/$domain/$f.$tgt
  python3 $WP_ENCODE --vocab $WP_VOCAB \
      --inputs $raw_outdir/$domain/$f.$src $raw_outdir/$domain/$f.$tgt \
      --outputs $raw_outdir/$domain/wp.$f.$src $raw_outdir/$domain/wp.$f.$tgt --min-len 5 --max-len 510
  # create label
  TRAIN_SIZE=`wc -l $raw_outdir/$domain/wp.$f.$src | cut -d' ' -f1 `
  yes 0 | head -n $TRAIN_SIZE > $raw_outdir/$domain/$f.label

  cat $raw_news_dir/valid.$src | sed 's/ //g' | sed 's/▁/ /g' > $raw_outdir/$domain/$f.valid.$src
  cat $raw_news_dir/valid.$tgt | sed 's/ //g' | sed 's/▁/ /g' > $raw_outdir/$domain/$f.valid.$tgt
  python3 $WP_ENCODE --vocab $WP_VOCAB \
      --inputs $raw_outdir/$domain/$f.valid.$src $raw_outdir/$domain/$f.valid.$tgt \
      --outputs $raw_outdir/$domain/wp.$f.valid.$src $raw_outdir/$domain/wp.$f.valid.$tgt --min-len 5 --max-len 510
  TRAIN_SIZE=`wc -l $raw_outdir/$domain/wp.$f.valid.$src | cut -d' ' -f1 `
  yes 0 | head -n $TRAIN_SIZE > $raw_outdir/$domain/$f.valid.label


  # process train - positive
  f=positive
  shuf -n 200000 $raw_domain_dir/train.$src | sed 's/ //g' | sed 's/▁/ /g'  > $raw_outdir/$domain/$f.$src
  shuf -n 200000 $raw_domain_dir/train.$tgt | sed 's/ //g' | sed 's/▁/ /g'  > $raw_outdir/$domain/$f.$tgt
  python3 $WP_ENCODE --vocab $WP_VOCAB \
      --inputs $raw_outdir/$domain/$f.$src $raw_outdir/$domain/$f.$tgt \
      --outputs $raw_outdir/$domain/wp.$f.$src $raw_outdir/$domain/wp.$f.$tgt --min-len 5 --max-len 510
  # create label
  TRAIN_SIZE=`wc -l $raw_outdir/$domain/wp.$f.$src | cut -d' ' -f1 `
  yes 1 | head -n $TRAIN_SIZE > $raw_outdir/$domain/$f.label

  cat $raw_domain_dir/valid.$src | sed 's/ //g' | sed 's/▁/ /g' > $raw_outdir/$domain/$f.valid.$src
  cat $raw_domain_dir/valid.$tgt | sed 's/ //g' | sed 's/▁/ /g' > $raw_outdir/$domain/$f.valid.$tgt
  python3 $WP_ENCODE --vocab $WP_VOCAB \
      --inputs $raw_outdir/$domain/$f.valid.$src $raw_outdir/$domain/$f.valid.$tgt \
      --outputs $raw_outdir/$domain/wp.$f.valid.$src $raw_outdir/$domain/wp.$f.valid.$tgt --min-len 5 --max-len 510
  TRAIN_SIZE=`wc -l $raw_outdir/$domain/wp.$f.valid.$src | cut -d' ' -f1 `
  yes 1 | head -n $TRAIN_SIZE > $raw_outdir/$domain/$f.valid.label

  # combine data
  for l in $src $tgt; do
    cat $raw_outdir/$domain/wp.negative.$l \
      $raw_outdir/$domain/wp.positive.$l \
      > $raw_outdir/$domain/wp.train.$l
    cat $raw_outdir/$domain/wp.negative.valid.$l \
      $raw_outdir/$domain/wp.positive.valid.$l \
      > $raw_outdir/$domain/wp.valid.$l
  done
  cat $raw_outdir/$domain/negative.label \
      $raw_outdir/$domain/positive.label \
      > $raw_outdir/$domain/train.label
  cat $raw_outdir/$domain/negative.valid.label \
    $raw_outdir/$domain/positive.valid.label \
    > $raw_outdir/$domain/valid.label

  # binarize data
  for l in $src $tgt; do
    fairseq-preprocess \
      --only-source \
      --srcdict $SRC_VOCAB \
      --trainpref $raw_outdir/$domain/wp.train.$l \
      --validpref $raw_outdir/$domain/wp.valid.$l \
      --destdir $bin_outdir/$domain/$l/input0 \
      --workers 20
    fairseq-preprocess \
    --only-source \
    --trainpref $raw_outdir/$domain/train.label \
    --validpref $raw_outdir/$domain/valid.label \
    --destdir $bin_outdir/$domain/$l/label/ \
    --workers 20
  done

  echo "====================================="
done
