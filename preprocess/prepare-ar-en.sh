#!/bin/bash

SCRIPTS=mosesdecoder/scripts
TOKENIZER=$SCRIPTS/tokenizer/tokenizer.perl
NORM_PUNC=$SCRIPTS/tokenizer/normalize-punctuation.perl
REM_NON_PRINT_CHAR=$SCRIPTS/tokenizer/remove-non-printing-char.perl

TGT_LANG=ar
DATASET="GlobalVoices News-Commentary UN WikiMatrix UNPC"
mkdir -p news_en_$TGT_LANG && mkdir -p news_en_$TGT_LANG/tmp  && cd news_en_$TGT_LANG/tmp && \
    opus_express -s $TGT_LANG -t en -c $DATASET --download-dir orig --test-quota 2000 --dev-quota 2000 -q 2>&1 \
    | tee news_en_$TGT_LANG.log
src=en
tgt=ar
lang=en-ar
prep=news_en_ar
tmp=$prep/tmp
orig=$tmp/orig

mkdir -p $orig $tmp $prep
cat $tmp/train.en | \
    perl $NORM_PUNC en | \
    perl $REM_NON_PRINT_CHAR | \
    perl $TOKENIZER -threads 8 -a -l en >> $tmp/train.tok.en

cat $tmp/dev.en | \
    perl $NORM_PUNC en | \
    perl $REM_NON_PRINT_CHAR | \
    perl $TOKENIZER -threads 8 -a -l en >> $tmp/valid.tok.en

cat $tmp/test.en | perl $TOKENIZER -threads 8 -a -l en > $tmp/test.tok.en


for l in $src $tgt; do
    rm $tmp/train.tags.$lang.tok.$l
    for f in "${CORPORA[@]}"; do
        cat $orig/$f.$l | \
            perl $NORM_PUNC $l | \
            perl $REM_NON_PRINT_CHAR | \
            perl $TOKENIZER -threads 8 -a -l $l >> $tmp/train.tags.$lang.tok.$l
    done
done

# preprocess en with moses, arabic with camel

camel_arclean $tmp/train.ar | camel_word_tokenize -o $tmp/train.tok.ar
camel_arclean $tmp/dev.ar | camel_word_tokenize -o $tmp/valid.tok.ar
camel_arclean $tmp/test.ar | camel_word_tokenizer -o $tmp/test.tok.ar

perl $CLEAN -ratio 2 $tmp/train.tags.$lang.tok $src $tgt $tmp/train.tok 1 175

cat $tmp/train.tok.en $tmp/valid.tok.en > $tmp/train.sp.en
cat $tmp/train.tok.ar $tmp/valid.tok.ar > $tmp/train.sp.ar

# learn sentence piece
SP_MODEL_TYPE=unigram
AR_SP_MODEL_NAME=sp_ar_en_unigram-ar
EN_SP_MODEL_NAME=sp_ar_en_unigram-en
BPEROOT=subword-nmt/subword_nmt
BPE_TOKENS=40000
VOCAB_SIZE=32000

TRAIN=$tmp/train.sp
echo "learn sentencepiece ${SP_MODEL_TYPE} ${TRAIN}..."
python3 ../scripts/spm_train.py --input=$TRAIN.en --model_prefix=$EN_SP_MODEL_NAME \
  --shuffle_input_sentence=true --input_sentence_size=5000000 \
  --vocab_size=$VOCAB_SIZE --character_coverage=1.0 --model_type=$SP_MODEL_TYPE 2>&1 | tee $EN_SP_MODEL_NAME.log

python3 ../scripts/spm_train.py --input=$TRAIN.ar --model_prefix=$AR_SP_MODEL_NAME \
  --shuffle_input_sentence=true --input_sentence_size=5000000 \
  --vocab_size=$VOCAB_SIZE --character_coverage=0.9995 --model_type=$SP_MODEL_TYPE 2>&1 | tee $AR_SP_MODEL_NAME.log

for f in train valid test; do
    echo "encode sentencepiece to ${f}..."
    python3 ../scripts/spm_encode_separate.py --models ${AR_SP_MODEL_NAME}.model ${EN_SP_MODEL_NAME}.model \
        --inputs $tmp/$f.tok.ar $tmp/$f.tok.en \
        --outputs $tmp/sp.$f.ar $tmp/sp.$f.en
done

for L in $src $tgt; do
    cp $tmp/sp.test.$L $prep/test.$L
    cp $tmp/sp.train.$L $prep/train.$L
    cp $tmp/sp.valid.$L $prep/valid.$L
done

echo "prepare vocab"
cat $EN_SP_MODEL_NAME.vocab | cut -d$'\t' -f1 | \
  sed 's/$/ 1/g' | \
  sed 's/^<unk> 1/<unk> 1 #fairseq:overwrite/g' | \
  sed 's/^<s> 1/<s> 1 #fairseq:overwrite/g' | \
  sed 's/^<\/s> 1/<\/s> 1 #fairseq:overwrite/g' > $EN_SP_MODEL_NAME.vocab.txt

cat $AR_SP_MODEL_NAME.vocab | cut -d$'\t' -f1 | \
  sed 's/$/ 1/g' | \
  sed 's/^<unk> 1/<unk> 1 #fairseq:overwrite/g' | \
  sed 's/^<s> 1/<s> 1 #fairseq:overwrite/g' | \
  sed 's/^<\/s> 1/<\/s> 1 #fairseq:overwrite/g' > $AR_SP_MODEL_NAME.vocab.txt