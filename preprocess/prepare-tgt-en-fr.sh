#!/bin/bash

src=fr
tgt=en
lang=en-fr

SCRIPTS=mosesdecoder/scripts
TOKENIZER=$SCRIPTS/tokenizer/tokenizer.perl
CLEAN=$SCRIPTS/training/clean-corpus-n.perl
NORM_PUNC=$SCRIPTS/tokenizer/normalize-punctuation.perl
REM_NON_PRINT_CHAR=$SCRIPTS/tokenizer/remove-non-printing-char.perl
SP_MODEL_TYPE=unigram
SP_MODEL_NAME=sp_fr_en_unigram

DATASETS=( "law" "med" "it" "koran" )

prep="law_en_${src}"
tmp=$prep/tmp
orig=$tmp/orig
mkdir -p $prep $tmp $orig

cd $orig
wget https://object.pouta.csc.fi/OPUS-JRC-Acquis/v3.0/moses/en-fr.txt.zip
unzip en-fr.txt.zip && rm en-fr.txt.zip
cd ../../..
echo "pre-processing train data..."
for l in $src $tgt; do
    rm $tmp/train.tags.$lang.tok.$l
    f="JRC-Acquis.en-fr"
    cat $orig/$f.$l | \
        perl $NORM_PUNC $l | \
        perl $REM_NON_PRINT_CHAR | \
        perl $TOKENIZER -threads 8 -a -l $l >> $tmp/train.tags.$lang.tok.$l
done

############################3
prep="med_en_${src}"
tmp=$prep/tmp
orig=$tmp/orig
mkdir -p $prep $tmp $orig
cd $orig
wget https://object.pouta.csc.fi/OPUS-EMEA/v3/moses/en-fr.txt.zip
unzip en-fr.txt.zip && rm en-fr.txt.zip
echo "pre-processing train data..."
cd ../../..
for l in $src $tgt; do
    rm $tmp/train.tags.$lang.tok.$l
    f="EMEA.en-fr"
    cat $orig/$f.$l | \
        perl $NORM_PUNC $l | \
        perl $REM_NON_PRINT_CHAR | \
        perl $TOKENIZER -threads 8 -a -l $l >> $tmp/train.tags.$lang.tok.$l
done

#############################
prep="it_en_${src}"
tmp=$prep/tmp
orig=$tmp/orig
mkdir -p $prep $tmp $orig
cd $orig
wget https://object.pouta.csc.fi/OPUS-GNOME/v1/moses/en-fr.txt.zip
unzip en-fr.txt.zip && rm en-fr.txt.zip README LICENSE
echo "pre-processing train data..."
for l in $src $tgt; do
    rm $tmp/train.tags.$lang.tok.$l
    f="GNOME.en-fr"
    cat $orig/$f.$l | \
        perl $NORM_PUNC $l | \
        perl $REM_NON_PRINT_CHAR | \
        perl $TOKENIZER -threads 8 -a -l $l >> $tmp/train.tags.$lang.tok.$l
done
wget https://object.pouta.csc.fi/OPUS-KDE4/v2/moses/en-fr.txt.zip
unzip en-fr.txt.zip && rm en-fr.txt.zip README LICENSE
echo "pre-processing train data..."
for l in $src $tgt; do
    rm $tmp/train.tags.$lang.tok.$l
    f="KDE4.en-fr"
    cat $orig/$f.$l | \
        perl $NORM_PUNC $l | \
        perl $REM_NON_PRINT_CHAR | \
        perl $TOKENIZER -threads 8 -a -l $l >> $tmp/train.tags.$lang.tok.$l
done
wget https://object.pouta.csc.fi/OPUS-Ubuntu/v14.10/moses/en-fr.txt.zip
unzip en-fr.txt.zip && rm en-fr.txt.zip README LICENSE
echo "pre-processing train data..."
for l in $src $tgt; do
    rm $tmp/train.tags.$lang.tok.$l
    f="Ubuntu.en-fr"
    cat $orig/$f.$l | \
        perl $NORM_PUNC $l | \
        perl $REM_NON_PRINT_CHAR | \
        perl $TOKENIZER -threads 8 -a -l $l >> $tmp/train.tags.$lang.tok.$l
done
wget https://object.pouta.csc.fi/OPUS-PHP/v1/moses/en-fr.txt.zip
unzip en-fr.txt.zip && rm en-fr.txt.zip README LICENSE
echo "pre-processing train data..."
for l in $src $tgt; do
    rm $tmp/train.tags.$lang.tok.$l
    f="PHP.en-fr"
    cat $orig/$f.$l | \
        perl $NORM_PUNC $l | \
        perl $REM_NON_PRINT_CHAR | \
        perl $TOKENIZER -threads 8 -a -l $l >> $tmp/train.tags.$lang.tok.$l
done

perl $CLEAN -ratio 1.5 $tmp/train.tags.$lang.tok $src $tgt $tmp/train.tags.$lang.clean 1 175



################################
cd ../..
prep="koran_en_${src}"
tmp=$prep/tmp
orig=$tmp/orig
mkdir -p $prep $tmp $orig
cd $orig
wget https://object.pouta.csc.fi/OPUS-Tanzil/v1/moses/en-fr.txt.zip
unzip en-fr.txt.zip && rm en-fr.txt.zip README LICENSE
cd ../../..
echo "pre-processing train data..."
for l in $src $tgt; do
    rm $tmp/train.tags.$lang.tok.$l
    f="Tanzil.en-fr"
    cat $orig/$f.$l | \
        perl $NORM_PUNC $l | \
        perl $REM_NON_PRINT_CHAR | \
        perl $TOKENIZER -threads 8 -a -l $l >> $tmp/train.tags.$lang.tok.$l
done

