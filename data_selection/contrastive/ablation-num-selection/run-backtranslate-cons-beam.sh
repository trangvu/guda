#!/bin/bash
ALNMT_DIR='/home/trangvu/uda-nmt/alnmt'
ROOT_DIR=$ALNMT_DIR
module load python37
module load cuda-11.2.0-gcc-10.2.0-gsjevs3
source /home/trangvu/uda-nmt/env/bin/activate

LANG=1
#de
DOMAIN=$1
K=$2
M=$3
#for DOMAIN in law ted it med koran; do
#  for K in 1 2 3 5 7 10; do
#    for M in 10000 50000 100000 200000 500000 1000000; do
GENSUBSET=cons-k.${K}-m.${M}
./do_backtranslate-beam.sh $LANG $DOMAIN $GENSUBSET
#    done
#  done
#done
