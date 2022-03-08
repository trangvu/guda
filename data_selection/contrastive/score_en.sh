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
langpair=$src-$tgt
domain=$2
k=$3

echo "eval model on $domain"
model_dir=$ALNMT_DIR/nmt/acl2021/models/domain_disc_cons/$langpair-k.${k}/$domain/$tgt
databin=$ALNMT_DIR/nmt/acl2021/preprocess/data-bin/domain_disc/$langpair/$domain/$tgt
outdir=$ALNMT_DIR/nmt/acl2021/preprocess/cons_selection/$langpair-k.${k}/$domain/$src
mkdir -p $outdir
USR_DIR=$ALNMT_DIR/nmt/acl2021/cidds

mono_dir=$ALNMT_DIR/nmt/acl2021/preprocess/monolingual/mono.10m.shards.$src
for SHARD in $(seq -f "%02g" 0 9); do \
  output=score.${SHARD}.txt
  echo '**************************************'
  echo 'Score discriminator on en for domain '$domain
  echo 'DATA_DIR  : '$mono_dir
  echo 'PATH      : '$model_dir
  echo 'OUTPUT    : '$outdir/$output
  echo '**************************************'

  python3 score.py  \
   --model-path $model_dir/checkpoint_best.pt \
   --score-input $mono_dir/mono${SHARD}.$src \
   --score-output $outdir/$output \
   --user-dir $USR_DIR \
   --data $databin
done