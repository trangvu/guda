#!/bin/bash
# Adapted from https://github.com/facebookresearch/MIXER/blob/master/prepareData.sh

#echo 'Cloning Moses github repository (for tokenization scripts)...'
#git clone https://github.com/moses-smt/mosesdecoder.git
#
#echo 'Cloning Subword NMT repository (for BPE pre-processing)...'
#git clone https://github.com/rsennrich/subword-nmt.git

SCRIPTS=mosesdecoder/scripts
TOKENIZER=$SCRIPTS/tokenizer/tokenizer.perl
CLEAN=$SCRIPTS/training/clean-corpus-n.perl
NORM_PUNC=$SCRIPTS/tokenizer/normalize-punctuation.perl
REM_NON_PRINT_CHAR=$SCRIPTS/tokenizer/remove-non-printing-char.perl

SP_MODEL_TYPE=unigram
SP_MODEL_NAME=sp_fi_en_unigram
VOCAB_SIZE=32000

BPEROOT=subword-nmt/subword_nmt
BPE_TOKENS=40000

URLS=(
    "http://www.statmt.org/europarl/v9/training/europarl-v9.fi-en.tsv.gz"
    "https://s3.amazonaws.com/web-language-models/paracrawl/release3/en-fi.bicleaner07.tmx.gz"
    "http://data.statmt.org/wikititles/v1/wikititles-v1.fi-en.tsv.gz"
    "https://tilde-model.s3-eu-west-1.amazonaws.com/rapid2016.en-fi.tmx.zip"
    "http://data.statmt.org/wmt19/translation-task/dev.tgz"
    "http://data.statmt.org/wmt19/translation-task/test.tgz"
)
FILES=(
    "europarl-v9.fi-en.tsv.gz"
    "en-fi.bicleaner07.tmx.gz"
    "wikititles-v1.fi-en.tsv.gz"
    "rapid2016.en-fi.tmx.zip"
    "dev.tgz"
    "test.tgz"
)

FI_EN_TSV_CORPORA=(
    "europarl-v9.fi-en.tsv"
    "wikititles-v1.fi-en.tsv"
)

EN_FI_TSV_CORPORA=(
    "en-fi.bicleaner07.tmx"
    "rapid2016.en-fi.tmx"
)

CORPORA=(
  "europarl-v10.de-en"
  "en-de"
  "commoncrawl.de-en"
  "rapid_2019.de-en"
  "news-commentary-v15.de-en"
  "wikititles-v2.de-en"
  "WikiMatrix.v1.de-en.langid"
)


if [ ! -d "$SCRIPTS" ]; then
    echo "Please set SCRIPTS variable correctly to point to Moses scripts."
    exit
fi

src=en
tgt=fi
lang=en-fi
prep=wmt19_fi_en
tmp=$prep/tmp
orig=wmt19orig
dev=dev/newstest2018

mkdir -p $orig $tmp $prep

cd $orig

for ((i=0;i<${#URLS[@]};++i)); do
    file=${FILES[i]}
    if [ -f $file ]; then
        echo "$file already exists, skipping download"
    else
        url=${URLS[i]}
        wget "$url"
        if [ -f $file ]; then
            echo "$url successfully downloaded."
        else
            echo "$url not successfully downloaded."
            exit -1
        fi
        if [ ${file: -4} == ".tgz" ]; then
            tar zxvf $file
        elif [ ${file: -4} == ".tar" ]; then
            tar xvf $file
        elif [ ${file: -3} == ".gz" ]; then
            gunzip $file
        elif [ ${file: -4} == ".zip" ]; then
            unzip $file
        fi
    fi
done
cd ..

echo "pre-process fi-en tsv parallel dataset"
for f in "${FI_EN_TSV_CORPORA[@]}"; do
  f_org="${f%.*}"
  cat $orig/$f | \
      cut -d$'\t' -f1 > $orig/$f_org.fi
  cat $orig/$f | \
      cut -d$'\t' -f2 > $orig/$f_org.en
done

echo "pre-process en-fi tsv parallel dataset"
for f in "${EN_FI_TSV_CORPORA[@]}"; do
  f_org="${f%.*}"
  cat $orig/$f | \
      cut -d$'\t' -f1 > $orig/$f_org.en
  cat $orig/$f | \
      cut -d$'\t' -f2 > $orig/$f_org.fi
done

echo "pre-processing train data..."
for l in $src $tgt; do
    rm $tmp/train.tags.$lang.tok.$l
    for f in "${CORPORA[@]}"; do
        cat $orig/$f.$l | \
            perl $NORM_PUNC $l | \
            perl $REM_NON_PRINT_CHAR | \
            perl $TOKENIZER -threads 8 -a -l $l >> $tmp/train.tags.$lang.tok.$l
    done
done

echo "pre-processing test data..."
for l in $src $tgt; do
    if [ "$l" == "$src" ]; then
        t="src"
    else
        t="ref"
    fi
    grep '<seg id' $orig/test-full/newstest2014-deen-$t.$l.sgm | \
        sed -e 's/<seg id="[0-9]*">\s*//g' | \
        sed -e 's/\s*<\/seg>\s*//g' | \
        sed -e "s/\???/\'/g" | \
    perl $TOKENIZER -threads 8 -a -l $l > $tmp/test.$l
    echo ""
done

perl $CLEAN -ratio 1.5 $tmp/train.tags.$lang.tok $src $tgt $tmp/train.tags.$lang.clean 1 175

echo "splitting train and valid..."
for l in $src $tgt; do
    awk '{if (NR%1333 == 0)  print $0; }' $tmp/train.tags.$lang.clean.$l > $tmp/valid.$l
    awk '{if (NR%1333 != 0)  print $0; }' $tmp/train.tags.$lang.clean.$l > $tmp/train.$l
done

TRAIN=$tmp/train.fi-en
rm -f $TRAIN
for l in $src $tgt; do
    cat $tmp/train.$l >> $TRAIN
done

echo "learn sentencepiece ${$SP_MODEL_TYPE} ${TRAIN}..."
python3 ../scripts/spm_train.py --input=$TRAIN --model_prefix=$SP_MODEL_NAME \
  --vocab_size=$VOCAB_SIZE --character_coverage=1.0 --model_type=$SP_MODEL_TYPE 2>&1 | tee $SP_MODEL_NAME.log

for L in $src $tgt; do
    for f in train.$L valid.$L test.$L; do
        echo "encode sentencepiece to ${f}..."
        python3 ../scripts/spm_encode.py --model=${SP_MODEL_NAME}.model --inputs $tmp/$f --outputs $tmp/sp.$f
    done
done

for L in $src $tgt; do
    cp $tmp/sp.test.$L $prep/test.$L
done

echo "prepare vocab"
cat $SP_MODEL_NAME.vocab | cut -d$'\t' -f1 | \
  sed 's/$/ 1/g' | \
  sed 's/^<unk> 1/<unk> 1 #fairseq:overwrite/g' | \
  sed 's/^<s> 1/<s> 1 #fairseq:overwrite/g' | \
  sed 's/^<\/s> 1/<\/s> 1 #fairseq:overwrite/g' > $SP_MODEL_NAME.vocab.txt