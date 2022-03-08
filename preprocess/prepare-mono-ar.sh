#!/bin/bash

SCRIPTS=mosesdecoder/scripts
TOKENIZER=$SCRIPTS/tokenizer/tokenizer.perl
NORM_PUNC=$SCRIPTS/tokenizer/normalize-punctuation.perl
REM_NON_PRINT_CHAR=$SCRIPTS/tokenizer/remove-non-printing-char.perl


SUBSAMPLE_SIZE=50000000
LANG=ar
SP_MODEL_NAME=sp_${LANG}_en_unigram


OUTDIR=mono_$LANG
orig=orig
tmp=$OUTDIR/tmp
mkdir -p $OUTDIR $tmp


URLS=(
    "http://web-language-models.s3-website-us-east-1.amazonaws.com/ngrams/ar/deduped/ar.deduped.xz"
)
FILES=(
    "ar.deduped.xz"
)


cd $orig
for ((i=0;i<${#URLS[@]};++i)); do
    file=${FILES[i]}
    if [ -f $file ]; then
        echo "$file already exists, skipping download"
    else
        url=${URLS[i]}
        wget "$url"
    fi
done
cd ..


if [ -f $tmp/monolingual.${SUBSAMPLE_SIZE}.${LANG} ]; then
    echo "found monolingual sample, skipping shuffle/sample/tokenize"
else
    for f in "${FILES[@]}"; do
        xz --decompress $orig/$f | \
            perl $NORM_PUNC $LANG | \
            perl $REM_NON_PRINT_CHAR | \
            perl $TOKENIZER -threads 8 -a -l $LANG >> $tmp/monolingual.${SUBSAMPLE_SIZE}.${LANG}
    done
fi


if [ -f $tmp/sp.monolingual.${SUBSAMPLE_SIZE}.${LANG} ]; then
    echo "found sentence piece monolingual sample, skipping SP step"
else
    python3 ../scripts/spm_encode.py --model=${SP_MODEL_NAME}.model \
        --inputs $tmp/monolingual.${SUBSAMPLE_SIZE}.${LANG} \
        --outputs $tmp/sp.monolingual.${SUBSAMPLE_SIZE}.${LANG}
fi


if [ -f $tmp/sp.monolingual.dedup.${SUBSAMPLE_SIZE}.${LANG} ]; then
    echo "found deduplicated monolingual sample, skipping deduplication step"
else
    echo "deduplicating"
    awk '!X[$0]++' $tmp/sp.monolingual.${SUBSAMPLE_SIZE}.${LANG} > $tmp/sp.monolingual.dedup.${SUBSAMPLE_SIZE}.${LANG}
fi


if [ -f $OUTDIR/sp.monolingual.dedup.00.$LANG ]; then
    echo "found sharded data, skipping sharding step"
else
    split --lines 1000000 --numeric-suffixes \
        --additional-suffix .${LANG} \
        $tmp/sp.monolingual.dedup.${SUBSAMPLE_SIZE}.${LANG} \
        $OUTDIR/sp.monolingual.dedup.
fi
