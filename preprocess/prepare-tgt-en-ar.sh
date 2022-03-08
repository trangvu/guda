#!/bin/bash

src=ar
tgt=en
lang=en-ar

SCRIPTS=mosesdecoder/scripts
TOKENIZER=$SCRIPTS/tokenizer/tokenizer.perl
CLEAN=$SCRIPTS/training/clean-corpus-n.perl
NORM_PUNC=$SCRIPTS/tokenizer/normalize-punctuation.perl
REM_NON_PRINT_CHAR=$SCRIPTS/tokenizer/remove-non-printing-char.perl

DATASETS=( "it" "koran" "ted" "tico-19" )

for DATASET in "${DATASETS[@]}"; do

    prep="${DATASET}_en_${src}"
    tmp=$prep/tmp
    orig=$tmp/orig
    # tokenize $src=ar
    camel_arclean $orig/train.ar | camel_word_tokenize -o $tmp/train.tok.ar
        camel_arclean $orig/valid.ar | camel_word_tokenize -o $tmp/valid.tok.ar
    camel_arclean $orig/test.ar | camel_word_tokenize -o $tmp/test.ar


    # tokenize en
    f="train" && cat $orig/$f.$tgt | \
        perl $NORM_PUNC $tgt | \
        perl $REM_NON_PRINT_CHAR | \
        perl $TOKENIZER -threads 8 -a -l $tgt > $tmp/$f.tok.$tgt

    f="valid" && cat $orig/$f.$tgt | \
        perl $NORM_PUNC $tgt | \
        perl $REM_NON_PRINT_CHAR | \
        perl $TOKENIZER -threads 8 -a -l $tgt > $tmp/$f.tok.$tgt

    cat $orig/test.$tgt | perl $TOKENIZER -threads 8 -a -l $tgt > $tmp/test.$tgt
    perl $CLEAN -ratio 2 $tmp/train.tok $src $tgt $tmp/train 1 175
    perl $CLEAN -ratio 2 $tmp/valid.tok $src $tgt $tmp/valid 1 175
done