#!/bin/bash
ALNMT_DIR='/home/trangvu/uda-nmt/alnmt'
ROOT_DIR=$ALNMT_DIR
module load python37
module load cuda-11.2.0-gcc-10.2.0-gsjevs3
source /home/trangvu/uda-nmt/env/bin/activate

l=$1
MONO_DIR=$ALNMT_DIR/nmt/acl2021/preprocess/monolingual
WP_VOCAB=$ALNMT_DIR/nmt/acl2021/data_selection/contrastive/vocab.txt
SRC_VOCAB=$ALNMT_DIR/nmt/acl2021/data_selection/contrastive/distill_bert_vocab.txt
WP_ENCODE=$ALNMT_DIR/nmt/acl2021/scripts/wp_encode.py

outdir=$ALNMT_DIR/nmt/acl2021/preprocess/data-bin/mono_wp/$l
mkdir -p $outdir
indir=$MONO_DIR/mono.10m.shards.$l
for SHARD in $(seq -f "%02g" 0 9); do \
        f=mono${SHARD}.$l
        echo "Processing $f"
        python3 $WP_ENCODE --vocab $WP_VOCAB \
          --inputs $indir/$f \
          --outputs $indir/wp.$f

        # Binarize
        fairseq-preprocess \
          --only-source \
          --srcdict $SRC_VOCAB \
          --testpref $indir/wp.$f \
          --destdir $outdir/shard${SHARD} \
          --workers 20
done
