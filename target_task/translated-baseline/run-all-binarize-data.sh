#!/bin/bash
ALNMT_DIR='/home/trangvu/uda-nmt/alnmt'
ROOT_DIR=$ALNMT_DIR
module load python37
module load cuda-11.2.0-gcc-10.2.0-gsjevs3
source /home/trangvu/uda-nmt/env/bin/activate

LANG=de
for DOMAIN in law ted it med koran; do
    ./binarize-data.sh $DOMAIN $LANG
done

LANG=cs
for DOMAIN in law ted it med koran; do
    ./binarize-data.sh $DOMAIN $LANG
done

LANG=ar
for DOMAIN in ted it koran tico19; do
    ./binarize-data.sh $DOMAIN $LANG
done

LANG=fr
PREF=wmt14
for DOMAIN in law ted it med koran tico19; do
    ./binarize-data.sh $DOMAIN $LANG
done
