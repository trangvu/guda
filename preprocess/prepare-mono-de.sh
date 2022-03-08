#!/bin/bash

SCRIPTS=mosesdecoder/scripts
TOKENIZER=$SCRIPTS/tokenizer/tokenizer.perl
NORM_PUNC=$SCRIPTS/tokenizer/normalize-punctuation.perl
REM_NON_PRINT_CHAR=$SCRIPTS/tokenizer/remove-non-printing-char.perl


SUBSAMPLE_SIZE=50000000
LANG=de
SP_MODEL_NAME=sp_${LANG}_en_unigram


OUTDIR=mono_$LANG
orig=orig
tmp=$OUTDIR/tmp
mkdir -p $OUTDIR $tmp


URLS=(
    "http://data.statmt.org/news-crawl/de/news.2007.de.shuffled.deduped.gz"
    "http://data.statmt.org/news-crawl/de/news.2008.de.shuffled.deduped.gz"
    "http://data.statmt.org/news-crawl/de/news.2009.de.shuffled.deduped.gz"
    "http://data.statmt.org/news-crawl/de/news.2010.de.shuffled.deduped.gz"
    "http://data.statmt.org/news-crawl/de/news.2011.de.shuffled.deduped.gz"
    "http://data.statmt.org/news-crawl/de/news.2012.de.shuffled.deduped.gz"
    "http://data.statmt.org/news-crawl/de/news.2013.de.shuffled.deduped.gz"
    "http://data.statmt.org/news-crawl/de/news.2014.de.shuffled.deduped.gz"
    "http://data.statmt.org/news-crawl/de/news.2015.de.shuffled.deduped.gz"
    "http://data.statmt.org/news-crawl/de/news.2016.de.shuffled.deduped.gz"
    "http://data.statmt.org/news-crawl/de/news.2017.de.shuffled.deduped.gz"
    "http://data.statmt.org/news-crawl/de/news.2018.de.shuffled.deduped.gz"
    "http://data.statmt.org/news-crawl/de/news.2019.de.shuffled.deduped.gz"
)
FILES=(
    "news.2007.de.shuffled.deduped.gz"
    "news.2008.de.shuffled.deduped.gz"
    "news.2009.de.shuffled.deduped.gz"
    "news.2010.de.shuffled.deduped.gz"
    "news.2011.de.shuffled.deduped.gz"
    "news.2012.de.shuffled.deduped.gz"
    "news.2013.de.shuffled.deduped.gz"
    "news.2014.de.shuffled.deduped.gz"
    "news.2015.de.shuffled.deduped.gz"
    "news.2016.de.shuffled.deduped.gz"
    "news.2017.de.shuffled.deduped.gz"
    "news.2018.de.shuffled.deduped.gz"
    "news.2019.de.shuffled.deduped.gz"
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
        gzip -c -d $orig/$f | \
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
