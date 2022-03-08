#!/bin/bash

SCRIPTS=mosesdecoder/scripts
TOKENIZER=$SCRIPTS/tokenizer/tokenizer.perl
NORM_PUNC=$SCRIPTS/tokenizer/normalize-punctuation.perl
REM_NON_PRINT_CHAR=$SCRIPTS/tokenizer/remove-non-printing-char.perl

SUBSAMPLE_SIZE=50000000
LANG=en



OUTDIR=mono_$LANG
orig=orig
tmp=$OUTDIR/tmp
mkdir -p $OUTDIR $tmp


URLS=(
    "http://data.statmt.org/news-crawl/en/news.2007.en.shuffled.deduped.gz"
    "http://data.statmt.org/news-crawl/en/news.2008.en.shuffled.deduped.gz"
    "http://data.statmt.org/news-crawl/en/news.2009.en.shuffled.deduped.gz"
    "http://data.statmt.org/news-crawl/en/news.2010.en.shuffled.deduped.gz"
    "http://data.statmt.org/news-crawl/en/news.2011.en.shuffled.deduped.gz"
    "http://data.statmt.org/news-crawl/en/news.2012.en.shuffled.deduped.gz"
    "http://data.statmt.org/news-crawl/en/news.2013.en.shuffled.deduped.gz"
    "http://data.statmt.org/news-crawl/en/news.2014.en.shuffled.deduped.gz"
    "http://data.statmt.org/news-crawl/en/news.2015.en.shuffled.deduped.gz"
    "http://data.statmt.org/news-crawl/en/news.2016.en.shuffled.deduped.gz"
    "http://data.statmt.org/news-crawl/en/news.2017.en.shuffled.deduped.gz"
    "http://data.statmt.org/news-crawl/en/news.2018.en.shuffled.deduped.gz"
    "http://data.statmt.org/news-crawl/en/news.2019.en.shuffled.deduped.gz"
)
FILES=(
    "news.2007.en.shuffled.deduped.gz"
    "news.2008.en.shuffled.deduped.gz"
    "news.2009.en.shuffled.deduped.gz"
    "news.2010.en.shuffled.deduped.gz"
    "news.2011.en.shuffled.deduped.gz"
    "news.2012.en.shuffled.deduped.gz"
    "news.2013.en.shuffled.deduped.gz"
    "news.2014.en.shuffled.deduped.gz"
    "news.2015.en.shuffled.deduped.gz"
    "news.2016.en.shuffled.deduped.gz"
    "news.2017.en.shuffled.deduped.gz"
    "news.2018.en.shuffled.deduped.gz"
    "news.2019.en.shuffled.deduped.gz"
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

salloc awk 'NF>4 && NF<176' $tmp/monolingual.${SUBSAMPLE_SIZE}.${LANG} > $tmp/monolingual.clean.${LANG}
salloc terashuf < $tmp/monolingual.clean.${LANG} > $tmp/monolingual.shuf.clean.${LANG}
salloc awk '!X[$0]++' $tmp/monolingual.shuf.clean.${LANG} > $tmp/monolingual.dedup.clean.${LANG}
SP_MODEL_NAME=sp_${LANG}_en_unigram

salloc python3 ../scripts/spm_encode.py --model=${SP_MODEL_NAME}.model \
          --inputs $tmp/monolingual.dedup.clean.${LANG} \
          --outputs $tmp/sp.monolingual.${LANG}

for L in fr de cs ar; do
  tmp_l=$OUTDIR/tmp-$L
  out_l=$OUTDIR/out-$L
  mkdir -p $tmp_l $out_l
  SP_MODEL_NAME=sp_${L}_en_unigram
  if [ -f $tmp_l/sp.monolingual.${SUBSAMPLE_SIZE}.${LANG} ]; then
      echo "found sentence piece monolingual sample, skipping SP step"
  else

      python3 ../scripts/spm_encode.py --model=${SP_MODEL_NAME}.model \
          --inputs $tmp/monolingual.dedup.clean.${LANG} \
          --outputs $tmp_l/sp.$L-en.monolingual.${LANG}
  fi

  if [ -f $out_l/sp.monolingual.00.$LANG ]; then
    echo "found sharded data, skipping sharding step"
  else
    split --lines 1000000 --numeric-suffixes \
        --additional-suffix .${LANG} \
        $tmp_l/sp.$L-en.monolingual.${LANG} \
        $out_l/sp.$L-en.monolingual.
  fi
done
#
#if [ -f $tmp/sp.monolingual.dedup.${SUBSAMPLE_SIZE}.${LANG} ]; then
#    echo "found deduplicated monolingual sample, skipping deduplication step"
#else
#    echo "deduplicating"
#    awk '!X[$0]++' $tmp/sp.monolingual.${SUBSAMPLE_SIZE}.${LANG} > $tmp/sp.monolingual.dedup.${SUBSAMPLE_SIZE}.${LANG}
#fi



