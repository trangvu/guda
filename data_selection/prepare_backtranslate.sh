#!/bin/bash

ALNMT_DIR='/home/trangvu/uda-nmt/alnmt'
ROOT_DIR=$ALNMT_DIR
module load python37
module load cuda-11.2.0-gcc-10.2.0-gsjevs3
source /home/trangvu/uda-nmt/env/bin/activate

MODEL_DIR=$ALNMT_DIR/models
SAVE_DIR=$ALNMT_DIR/nmt/acl2021/data_selection/backtranslate

mkdir -p $SAVE_DIR

LANGUAGES=( fr de cs )
SRC_DATASETS=( "wmt14_en_fr" "wmt20_en_de" "wmt20_en_cs" )
DOMAINS=( law med it koran ted )

for ((i=0;i<${#LANGUAGES[@]};++i)); do
    L=${LANGUAGES[i]}
    src_dataset=${SRC_DATASETS[i]}
    src_models=( ${src_dataset}-from_en $src_dataset )
    src_langs=( en $L )
    tgt_langs=( $L en )
    for domain in "${DOMAINS[@]}"; do
        dataset=${domain}_en_${L}
        DATA_DIR=$ALNMT_DIR/nmt/acl2021/preprocess/selection/$dataset
        for ((j=0;j<${#src_models[@]};++i)); do
            m_dir=${src_models[j]}
            SRC_LANG=${src_langs[j]}
            TGT_LANG=${tgt_langs[j]}

            save_model_dir=${MODEL_DIR}/${m_dir}
            echo '**************************************'
            echo '[To-EN] Back transtion '$L'-en ON '$dataset
            echo 'SRC_LANG  : '$SRC_LANG
            echo 'TGT_LANG  : '$TGT_LANG
            echo 'DATA_DIR  : '$DATA_DIR
            echo 'SAVE_DIR  : '$save_model_dir
            echo '**************************************'
            mv $DATA_DIR/random_${L}.${L}-en.${L} $DATA_DIR/random.${L}-en.${L}

            for f in target random; do
                echo "Translate $f"
                log_name=gen_${domain}_${m_dir}_${f}
                fairseq-generate $DATA_DIR \
                    --path $save_model_dir/checkpoint_best.pt \
                    --dataset-impl raw \
                    --gen-subset $f \
                    --source-lang $SRC_LANG --target-lang $TGT_LANG \
                    --beam 5 --remove-bpe=sentencepiece --lenpen 0.6 | tee $SAVE_DIR/$log_name.log

                grep ^H $SAVE_DIR/$log_name.log | cut -f3 | sacremoses detokenize > $SAVE_DIR/$log_name.$TGT_LANG
                cp $SAVE_DIR/$log_name.$TGT_LANG $DATA_DIR/${f}1.${L}-en.$TGT_LANG
            done
#            cp $DATA_DIR/random_${TGT_LANG}1.${L}-en.$TGT_LANG $DATA_DIR/target1.${L}-en.$TGT_LANG
        done
    done
done