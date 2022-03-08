#!/bin/bash
ALNMT_DIR='/home/trangvu/uda-nmt/alnmt'
ROOT_DIR=$ALNMT_DIR
module load python37
module load cuda-11.2.0-gcc-10.2.0-gsjevs3
source /home/trangvu/uda-nmt/env/bin/activate

LANG=de
PREF=wmt20
sel_type=$1

for DOMAIN in law ted it med koran; do
    ./uda_joint_disc-src-avai.sh $DOMAIN $LANG ${PREF}_en_$LANG $sel_type
    ./uda_joint_disc-src-avai.sh $DOMAIN $LANG ${PREF}_en_$LANG target

    ./uda_joint_disc-tgt-avai.sh $DOMAIN $LANG ${PREF}_en_$LANG-from_en $sel_type
    ./uda_joint_disc-tgt-avai.sh $DOMAIN $LANG ${PREF}_en_$LANG-from_en target
done

LANG=cs
PREF=wmt20
for DOMAIN in law ted it med koran; do
    ./uda_joint_disc-src-avai.sh $DOMAIN $LANG ${PREF}_en_$LANG $sel_type
    ./uda_joint_disc-src-avai.sh $DOMAIN $LANG ${PREF}_en_$LANG target

    ./uda_joint_disc-tgt-avai.sh $DOMAIN $LANG ${PREF}_en_$LANG-from_en $sel_type
    ./uda_joint_disc-tgt-avai.sh $DOMAIN $LANG ${PREF}_en_$LANG-from_en target
done

LANG=ar
PREF=news
for DOMAIN in ted it koran; do
    ./uda_joint_disc-src-avai.sh $DOMAIN $LANG ${PREF}_en_$LANG $sel_type
    ./uda_joint_disc-src-avai.sh $DOMAIN $LANG ${PREF}_en_$LANG target

    ./uda_joint_disc-tgt-avai.sh $DOMAIN $LANG ${PREF}_en_$LANG-from_en $sel_type
    ./uda_joint_disc-tgt-avai.sh $DOMAIN $LANG ${PREF}_en_$LANG-from_en target
done

LANG=fr
PREF=wmt14
for DOMAIN in law ted it med koran; do
    ./uda_joint_disc-src-avai.sh $DOMAIN $LANG ${PREF}_en_$LANG $sel_type
    ./uda_joint_disc-src-avai.sh $DOMAIN $LANG ${PREF}_en_$LANG target

    ./uda_joint_disc-tgt-avai.sh $DOMAIN $LANG ${PREF}_en_$LANG-from_en $sel_type
    ./uda_joint_disc-tgt-avai.sh $DOMAIN $LANG ${PREF}_en_$LANG-from_en target
done
