#!/bin/bash

ALNMT_DIR='/home/xvuthith/da33_scratch/trang/uda-nmt/alnmt'
ROOT_DIR=$ALNMT_DIR
module load python/3.7.2-gcc6
module load cuda/10.1
module load cudnn/7.6.5-cuda10.1
source $HOME/da33_scratch/trang/newenv/bin/activate


MODEL_DIR=$ALNMT_DIR/models


index=$1
domain=$2
LANGUAGES=( fr de cs )
SRC_DATASETS=( "wmt14_en_fr" "wmt20_en_de" "wmt20_en_cs" )
DOMAINS=( law med it koran ted )

L=${LANGUAGES[index]}
src_dataset=${SRC_DATASETS[index]}
src_models=( ${src_dataset}-from_en $src_dataset )
src_langs=( en $L )
tgt_langs=( $L en )


dataset=${domain}_en_${L}
DATA_DIR=$ALNMT_DIR/nmt/acl2021/preprocess/selection/$dataset
SAVE_DIR=$ALNMT_DIR/nmt/acl2021/preprocess/selection/$dataset/backtranslate
mkdir -p $SAVE_DIR
for ((j=0;j<${#src_models[@]};++i)); do
    m_dir=${src_models[j]}
    SRC_LANG=${src_langs[j]}
    TGT_LANG=${tgt_langs[j]}

    save_model_dir=${MODEL_DIR}/${m_dir}
    echo '**************************************'
    echo 'Back transtion '$SRC_LANG'-'$TGT_LANG' ON '$dataset
    echo 'SRC_LANG  : '$SRC_LANG
    echo 'TGT_LANG  : '$TGT_LANG
    echo 'DATA_DIR  : '$DATA_DIR
    echo 'SAVE_DIR  : '$save_model_dir
    echo '**************************************'

    for f in target_$SRC_LANG random_$SRC_LANG ; do
        echo "Translate $f"
        log_name=gen_${domain}_${m_dir}_${f}
        fairseq-generate $DATA_DIR \
            --path $save_model_dir/checkpoint_best.pt \
            --dataset-impl raw \
            --batch-size 128 \
            --gen-subset $f \
            --source-lang $SRC_LANG --target-lang $TGT_LANG \
            --beam 5 --remove-bpe=sentencepiece --lenpen 0.6 | tee $SAVE_DIR/$log_name.log

        grep ^H $SAVE_DIR/$log_name.log | cut -f3 | sacremoses detokenize > $SAVE_DIR/$log_name.$TGT_LANG
        cp $SAVE_DIR/$log_name.$TGT_LANG $DATA_DIR/${f}1.${L}-en.$TGT_LANG
    done
done