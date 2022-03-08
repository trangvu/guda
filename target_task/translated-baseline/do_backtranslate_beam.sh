#!/bin/bash
ALNMT_DIR='/home/trangvu/uda-nmt/alnmt'
ROOT_DIR=$ALNMT_DIR
module load python37
module load cuda-11.2.0-gcc-10.2.0-gsjevs3
source /home/trangvu/uda-nmt/env/bin/activate
MODEL_DIR=$ALNMT_DIR/models


index=$1
domain=$2
gensubset=$3
LANGUAGES=( fr de cs ar )
SRC_DATASETS=( "wmt14_en_fr" "wmt20_en_de" "wmt20_en_cs" "news_en_ar")

L=${LANGUAGES[index]}
src_dataset=${SRC_DATASETS[index]}
src_model=$src_dataset


dataset=${domain}_en_${L}
DATA_DIR=$ALNMT_DIR/nmt/acl2021/preprocess/data-bin/$dataset
SAVE_DIR=$ALNMT_DIR/nmt/acl2021/preprocess/selection/$dataset/backtranslate_beam
echo "SAVE DIR "$SAVE_DIR
tmp=$SAVE_DIR/tmp
mkdir -p $SAVE_DIR $tmp
suffix=$L-en

m_dir=$src_model
SRC_LANG=$L
TGT_LANG=en

save_model_dir=${MODEL_DIR}/${m_dir}
    echo '**************************************'
    echo 'Back transtion '$SRC_LANG'-'$TGT_LANG' ON '$dataset
    echo 'SRC_LANG  : '$SRC_LANG
    echo 'TGT_LANG  : '$TGT_LANG
    echo 'DATA_DIR  : '$DATA_DIR
    echo 'SAVE_DIR  : '$save_model_dir
    echo '**************************************'


    f="${gensubset}"
    filename=$f.$suffix.$SRC_LANG
    echo "Translate $filename"
    log_name=gen_${domain}_${m_dir}_${f}
    fairseq-generate $DATA_DIR \
        --path $save_model_dir/checkpoint_best.pt \
        --skip-invalid-size-inputs-valid-test \
        --max-tokens 32000 \
        --gen-subset $f \
        --source-lang $SRC_LANG --target-lang $TGT_LANG \
        --beam 5 --remove-bpe=sentencepiece --lenpen 0.6  >  $tmp/$log_name.log

    echo "Extract BT data and save to "$SAVE_DIR/bt_src.${SRC_LANG}_hyp.en_$f
    python3 extract_bt_data.py \
    --minlen 1 --maxlen 250 --ratio 1.5 \
    --output $SAVE_DIR/bt_src.${SRC_LANG}_hyp.en_$f --srclang $TGT_LANG --tgtlang $SRC_LANG \
    $tmp/$log_name.log
