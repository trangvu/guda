#!/bin/bash

ALNMT_DIR='/home/trangvu/uda-nmt/alnmt'
ROOT_DIR=$ALNMT_DIR
module load python37
module load cuda-11.2.0-gcc-10.2.0-gsjevs3
source /home/trangvu/uda-nmt/env/bin/activate

src=en
tgt=$1
langpair=$src-$tgt

echo "eval general model"
model_dir=$ALNMT_DIR/models/
model=lm-en$tgt-news
databin=$ALNMT_DIR/nmt/acl2021/preprocess/data-bin/mono/$langpair
outdir=$ALNMT_DIR/nmt/acl2021/preprocess/selection/$langpair/news
for SHARD in $(seq -f "%02g" 0 9); do \
  datadir=$databin/shard${SHARD}
  gensubset=test.$src-$tgt.$src
  output=shard${SHARD}/score.$gensubset.txt
  echo '**************************************'
  echo 'Score language model on dataset '$DATASET
  echo 'DATA_DIR  : '$datadir
  echo 'PATH      : '$model
  echo 'GENSUBSET : '$gensubset
  echo 'OUTPUT    : '$outdir/$output
  echo '**************************************'

  python3 score_entropy.py $datadir \
   --path $model_dir/$model/checkpoint_best.pt \
   --task language_modeling \
   --gen-subset $gensubset \
   --max-tokens 16000 \
   --score-output $outdir/$output \
   --skip-invalid-size-inputs-valid-test \
   --sample-break-mode "eos" --shorten-method truncate

  gensubset=test.$tgt-$src.$tgt
  output=shard${SHARD}/score.$gensubset.txt
  echo '**************************************'
  echo 'Score language model on dataset '$DATASET
  echo 'DATA_DIR  : '$datadir
  echo 'PATH      : '$model
  echo 'GENSUBSET : '$gensubset
  echo 'OUTPUT    : '$outdir/$output
  echo '**************************************'

  python3 score_entropy.py $datadir \
   --path $model_dir/$model/checkpoint_best.pt \
   --task language_modeling \
   --gen-subset $gensubset \
   --max-tokens 16000 \
   --score-output $outdir/$output \
   --skip-invalid-size-inputs-valid-test \
   --sample-break-mode "eos" --shorten-method truncate
done


for domain in law it med ted koran; do
  echo "Score indomain model $domain"
  # model train on $src -> score on tgt
  model=lm-koran_${src}_${tgt}-$src
  gensubset=test.$tgt-$src.$tgt
  outdir=$ALNMT_DIR/nmt/acl2021/preprocess/selection/$langpair/$domain
  mkdir -p $outdir
  for SHARD in $(seq -f "%02g" 0 9); do \
    output=shard${SHARD}/score.$gensubset.txt
    echo '**************************************'
    echo 'Score language model on dataset '$DATASET
    echo 'DATA_DIR  : '$datadir
    echo 'PATH      : '$model
    echo 'GENSUBSET : '$gensubset
    echo 'OUTPUT    : '$outdir/$output
    echo '**************************************'

    python3 score_entropy.py $datadir \
     --path $model_dir/$model/checkpoint_best.pt \
     --task language_modeling \
     --gen-subset $gensubset \
     --max-tokens 16000 \
     --score-output $outdir/$output \
     --skip-invalid-size-inputs-valid-test \
     --sample-break-mode "eos" --shorten-method truncate
    done

  # model train on $tgt -> score on $src
  model=lm-koran_${src}_${tgt}-$tgt
  gensubset=test.$src-$tgt.$src
  outdir=$ALNMT_DIR/nmt/acl2021/preprocess/selection/$langpair/$domain
  mkdir -p $outdir
  for SHARD in $(seq -f "%02g" 0 9); do \
    output=shard${SHARD}/score.$gensubset.txt
    echo '**************************************'
    echo 'Score language model on dataset '$DATASET
    echo 'DATA_DIR  : '$datadir
    echo 'PATH      : '$model
    echo 'GENSUBSET : '$gensubset
    echo 'OUTPUT    : '$outdir/$output
    echo '**************************************'

    python3 score_entropy.py $datadir \
     --path $model_dir/$model/checkpoint_best.pt \
     --task language_modeling \
     --gen-subset $gensubset \
     --max-tokens 16000 \
     --score-output $outdir/$output \
     --skip-invalid-size-inputs-valid-test \
     --sample-break-mode "eos" --shorten-method truncate
    done
done