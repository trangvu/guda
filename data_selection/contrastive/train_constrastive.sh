#!/bin/bash

ALNMT_DIR='/home/trangvu/uda-nmt/alnmt'
ROOT_DIR=$ALNMT_DIR
module load python37
module load cuda-11.2.0-gcc-10.2.0-gsjevs3
source /home/trangvu/uda-nmt/env/bin/activate

src=en
tgt=$1
lang_pair=en-$tgt
USR_DIR=$ALNMT_DIR/nmt/acl2021/cidds
echo "eval general model"
databin=$ALNMT_DIR/nmt/acl2021/preprocess/data-bin/constrastive/$lang_pair
outdir=$ALNMT_DIR/nmt/acl2021/models/contrastive
mkdir -p $outdir
k=$2
model=${lang_pair}-k.${k}
save_dir=$outdir/$model
#for k in 2 3 5 7 10; do
    echo "Train constrastive $k cluster for ${lang_pair}"
    echo '**************************************'
    echo 'DATA_DIR  : '$databin
    echo 'SAVE_DIR  : '$save_dir
    echo 'GENSUBSET : '$gensubset
    echo 'OUTPUT    : '$outdir/$output
    echo '**************************************'

fairseq-train $databin \
	--arch contrastive_distill_bert --task contrastive_nmt_from_pretrained_bert_task \
    --max-epoch 30 --max-sentences 128 \
    --user-dir $USR_DIR \
    --save-dir $save_dir \
    --source-lang $src --target-lang $tgt \
    --optimizer adam --adam-betas "(0.9, 0.98)" --clip-norm 0.0 \
    --lr 1e-3 --lr-scheduler inverse_sqrt --warmup-updates 5000 \
    --warmup-init-lr "1e-07" \
    --dropout 0.1 --weight-decay 0.0001 \
    --criterion contrastive_loss \
    --sentence-avg --update-freq 32 \
    --ddp-backend=no_c10d \
    --no-epoch-checkpoints \
    --precluster --labels label.$k --freeze-encoder \
    --max-source-positions 512 --max-target-positions 512 \
    --tensorboard-logdir $save_dir/log --disable-validation